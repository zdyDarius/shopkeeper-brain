import time
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

@node_log("node_item_name_confirm")
def node_item_name_confirm(state:QueryGraphState):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    logger.info("---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    session_id,original_query  = step_1_validate_and_get_data(state)

    history_list: list[dict] = step_2_get_history_by_session_id(session_id)

    llm_dict: dict[str, Any] = step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query)



    # 后面会调用大模型，进行逻辑处理
    # save_chat_message(
    #     session_id = state.get("session_id"),
    #     role = 'user',
    #     text = state.get("original_query"),
    #     rewritten_query="这是重写的问题",
    #     item_names=["商品A", "商品B"],
    #     image_urls=[]
    # )

    # time.sleep(1)
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    logger.info("---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state:QueryGraphState):
    session_id = state.get("session_id")
    original_query = state.get("original_query")

    if not session_id and not original_query:
        logger.error(f"session_id或者original_query为空,业务无法继续,提前终止!")
        raise ValueError(f"session_id或者original_query为空,业务无法继续,提前终止!")

    return session_id, original_query

@step_log("step_2_get_history_by_session_id")
def step_2_get_history_by_session_id(session_id):

    """
        _id:xx,
        session_id:xx,
        role:xx ,
        text:xx,
        rewritten_query:xx ,
        item_names:[],
        image_urls:[],
        ts
    """
    history_list = get_recent_messages(session_id)
    logger.debug(f"根据:{session_id}查询到的聊天记录:{len(history_list)}")
    # history_list = list(filter(lambda x: x.get('item_names'), history_list))
    history_list = [item for item in history_list if item.get("item_names")]
    logger.debug(f"根据:{session_id}过滤后的有效聊天记录:{len(history_list)}")
    return history_list

@step_log("step_3_call_llm_rewritten_and_extract_itemnames")
def step_3_call_llm_rewritten_and_extract_itemnames(history_list, original_query):


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

    prompt_text = load_prompt(name='rewritten_query_and_itemnames',history_text=history_text,query=original_query)


    llm_model = get_llm_client(json_mode=True)
    logger.debug(f"prompt_text: {prompt_text}")
    message = HumanMessage(content=prompt_text)

    chains = llm_model | JsonOutputParser()


    json_dict = chains.invoke([message])

    if not json_dict.get("item_names") :
        json_dict['item_names'] = []

    if not json_dict["rewritten_query"]:
        json_dict['rewritten_query'] = original_query

    logger.debug(f"json_dict: {json_dict}")

    return json_dict

if __name__ == "__main__":
    mock_state = {
        "session_id": "sess-owmlfnl0stmmue3rddz",
        "original_query": "我之前问你的商品到底怎么使用啊？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)