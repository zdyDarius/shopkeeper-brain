import time
import sys
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client,hybrid_search
from utils.clients.mongo_history_utils import save_chat_message, get_recent_messages
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体名称确认阈值：高于该分数 → 直接确认 [0.75]
ITEM_NAME_CONFIRM_THRESHOLD = 0.70
# 主体名称候选阈值：介于两者之间 → 让用户选择
ITEM_NAME_CANDIDATE_THRESHOLD = 0.60
# 给用户选择时，最多展示几个候选
ITEM_NAME_OPTIONS_TOPK = 2
ITEM_NAME_CONFIRMED_TOPK = 1
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

    milvus_dict = step_4_select_item_names_milvus(llm_dict.get("item_names"))

    confirmed_option_dict: dict[str, list[str]] = step_5_select_confirmed_and_option_item_names(milvus_dict)

    # TODO 7.根据确认和可选更新状态
    step_6_change_state_property(state, llm_dict.get("rewritten_query"), confirmed_option_dict)
    # TODO 8.向mongodb中写入对话
    step_7_save_user_chat_message(state)

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


@step_log("step_4_select_item_names_milvus")
def step_4_select_item_names_milvus(llm_item_names)->dict[str, list[dict]]:
    if not llm_item_names:
        return {}
    milvus_search_dict: dict[str, list[dict]] = {}
    result = generate_embeddings(llm_item_names)

    for index, llm_item_name in enumerate(llm_item_names):

        dense_vector = result.get("dense")[index]
        sparse_vector = result.get("sparse")[index]
        reqs_list = create_hybrid_search_requests(dense_vector, sparse_vector)

        milvus_client = get_milvus_client()
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs_list,
            ranker_weights=(0.4, 0.6),
            norm_score=True,  # 是否进行归一化处理
            output_fields=["item_name"]
        )
        real_response = response[0]
        llm_item_result_list: list[dict] = []
        for item in real_response:
            llm_item_result_list.append(
                {
                    "item_name":item.get("entity").get("item_name"),
                    "score": item.get("distance")
                 }
            )
        milvus_search_dict[llm_item_name] = llm_item_result_list
        logger.debug(f"milvus_search_dict: {milvus_search_dict}")
    return milvus_search_dict
@step_log("step_5_select_confirmed_and_option_item_names")
def step_5_select_confirmed_and_option_item_names(milvus_dict):
    """

        :param milvus_dict:
        {
        "商品A":[{"item_name":"商品a","score":0.7},{"item_name":"商品ab","score":0.6}]
        "商品B":[{"item_name":"商品b","score":0.7},{"item_name":"商品ab","score":0.6}]
        }
        :return:  {"confirmed_list":[item_name1],"option_list":[item_name2,...]}"}
        """
    if not milvus_dict:
        return {}
    confirmed_list: list[dict] = []
    option_list: list[dict] = []
    for llm_item_name,item_list in milvus_dict.items():
        high_list = [item.get('item_name') for item in item_list if item.get('score') >= ITEM_NAME_CONFIRM_THRESHOLD]
        middle_list = [item.get('item_name') for item in item_list if
                       ITEM_NAME_CANDIDATE_THRESHOLD  <=item.get('score',0.0) < ITEM_NAME_CONFIRM_THRESHOLD ]
        if high_list:
            confirmed_list.extend(
                high_list[:ITEM_NAME_CONFIRMED_TOPK]
            )
            logger.info(f"{llm_item_name}已经获取到可信的主体:{high_list[:ITEM_NAME_CONFIRMED_TOPK]}")
            continue
        if middle_list:
            confirmed_list.extend(
                middle_list[:ITEM_NAME_CONFIRMED_TOPK]
            )
            logger.info(f"{llm_item_name}没有可信的主体,但是有可选的主体:{middle_list[:ITEM_NAME_OPTIONS_TOPK]}")
    return {
        "confirmed_list":confirmed_list,
        "option_list":option_list
    }

@step_log("step_6_change_state_property")
def step_6_change_state_property(state,rewritten_querym, confirmed_option_dict):
    confirmed_list = confirmed_option_dict.get("confirmed_list")
    option_list = confirmed_option_dict.get("option_list")
    if confirmed_list:
        state['rewritten_query'] = rewritten_querym
        state['item_names'] = confirmed_list
        logger.info(f"已经有明确的item_name列表,已经更新state item_name以及rewritten_query属性!")
    elif option_list:
        state['answer'] = f"没有可信的item_name,但是有可选的:{option_list},请您明确!"
        logger.info(f"没有明确的item_name列表,但是有可选的!已经更新answer属性!")
    else:
        # 如果没有确认 也没有可选
        state['answer'] = f"没有可信,也没有可选的实体列表,请明确以后再次问题!!"
@step_log("step_7_change_state_property")
def step_7_save_user_chat_message(state):
    # 向mongodb中写入数据
    save_chat_message(
        session_id=state.get("session_id"),
        role="user",
        text=state.get("original_query"),
        rewritten_query=state.get("rewritten_query"),
        item_names=state.get("item_names")
    )




if __name__ == "__main__":
    mock_state = {
        "session_id": "sess-owmlfnl0stmmue3rddz",
        "original_query": "我之前问你的商品到底怎么使用啊？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)