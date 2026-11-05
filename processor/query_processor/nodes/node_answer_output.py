import re
import time
import sys

from langchain_core.messages import HumanMessage

from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.llm_client_txt_util import get_llm_client  # 文本大模型改走新工具（vllm/阿里云可切换）
from utils.load_prompt import load_prompt
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_running_task, add_done_task

@step_log('step_1_has_answer')
def step_1_has_answer(state):
    answer = state["answer"]
    return True if answer else False
@step_log('step_2_validate_and_get_data')
def step_2_validate_and_get_data(state:QueryGraphState):
    # reranked_docs, session_id, item_names, rewritten_query, is_stream
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names")
    session_id = state.get("session_id")
    is_stream = state.get("is_stream", False)
    rewritten_query = state.get("rewritten_query")
    if (not reranked_docs) or (not session_id) or (not item_names) or (not rewritten_query):
        logger.error("核心参数为空,业务无法继续,提前终止!")
        raise ValueError("核心参数为空,业务无法继续,提前终止!")
    return reranked_docs, session_id, item_names, rewritten_query, is_stream


@step_log('step_3_get_history_by_session_id')
def step_3_get_history_by_session_id(session_id):

    history_list = get_recent_messages(session_id)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # history_list = list(filter(lambda x: x.get('item_names'), history_list))
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    return history_list

@step_log('step_4_create_answer_prompt')
def step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query):
    context= ''
    for doc in history_list:
        context+=(f"标题:{doc.get('title')},数据类源:{'联网搜索' if doc.get('type') == 'web' else '向量数据库'}, "
                    f"置信度:{doc.get('score')},内容:{doc.get('text')} \n")
    history_text = ''
    if history_list:
        history_text_list = []
        for index, item in enumerate(history_list):
            if item.get("role") == "user":
                # 用户提问
                history_text_list.append(f"编号:{index},用户提问,原始问题:{item.get('text')},"
                                         f"重写后的问题:{item.get('rewritten_query')}，"
                                         f"关联item_name：{','.join(item.get('item_names'))}")
            else:
                # 助手回答
                history_text_list.append(f"编号:{index},助手回答,重写问题:{item.get('rewritten_query')},"
                                         f"回答结果:{item.get('text')[:50]},"
                                         f"关联item_name:{','.join(item.get('item_names'))}")
        history_text = '\n'.join(history_text_list)
    else:
        history_text = "当前会话id没有对应的历史记录"
    item_names_str = '/'.join(item_names)
    answer_prompt_text =load_prompt('answer_out',context=context,history=history_text,item_names=item_names_str,question=rewritten_query)
    return answer_prompt_text

@step_log('step_5_call_llm_create_answer')
def step_5_call_llm_create_answer(answer_prompt, is_stream, session_id):
    llm_clent = get_llm_client()

    message = HumanMessage(content = answer_prompt)
    answer: str = ""

    if is_stream:
        for chunk in llm_clent.stream([message]):
            chunk_str:str = chunk.content
            push_to_session(
                session_id=session_id,
                event=SSEEvent.DELTA,
                data={
                    'delta': chunk_str,
                }
            )

            answer +=  chunk_str
    else:
        res = llm_clent.invoke([message])
        answer = res.content

    return answer




@step_log('step_6_extract_chunk_and_url_image')
def step_6_extract_chunk_and_url_image(reranked_docs):
    image_urls = []
    rep = re.compile(r"!\[.*?]\((.*?)\)")

    for chunk in reranked_docs:
        url = chunk.get('url')
        text = chunk.get('text')
        if url:
            if url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                image_urls.append(url)
        if text:
            image_url_list = rep.findall(text)
            for image_url in image_url_list:
                if image_url and image_url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")):
                    image_urls.append(image_url)
    logger.info(f"完成切片中的图片提取,提取数量:{len(image_urls)},内容:{image_urls}")
    return image_urls



@step_log('step_7_save_assistant_message')
def step_7_save_assistant_message(state):
    save_chat_message(
        session_id=state.get("session_id"),
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names"),
        image_urls=state.get("image_urls")
    )


@node_log("node_answer_output")
def node_answer_output(state:QueryGraphState):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    #  1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #  2. 判断状态中是否有答案
    has_answer = step_1_has_answer(state)
    # 如果状态中没有答案，才会执行下面流程；如果有答案，直接写到历史记录中，结束节点执行
    if not has_answer:
        #  3. 从状态中获取相关业务数据并进行校验
        reranked_docs, session_id, item_names, rewritten_query, is_stream = step_2_validate_and_get_data(state)
        #  4. 根据session_id获取历史消息记录
        history_list = step_3_get_history_by_session_id(session_id)
        #  5. 拼接提示词
        answer_prompt: str = step_4_create_answer_prompt(reranked_docs, history_list, item_names, rewritten_query)
        #  6. 调用模型生成答案
        answer: str = step_5_call_llm_create_answer(answer_prompt, is_stream, session_id)
        #  7. 处理查询结果中的图片
        image_urls: list[str] = step_6_extract_chunk_and_url_image(reranked_docs)
        #  8. 更新状态
        state["answer"] = answer
        state["image_urls"] = image_urls

    #  9. 将助手消息保存到mongodb
    step_7_save_assistant_message(state)
    #  10. 添加节点到已完成列表
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.success("~~~全剧终~~~")
    return state

if __name__ == "__main__":
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)
            """,
        }
    ]
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？", "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法"},
    ]
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,
        "answer": None,
    }
    result = node_answer_output(mock_state)
    print(result)
