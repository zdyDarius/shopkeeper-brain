import time
import sys

from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from common.logging.logger import logger, node_log, step_log


@node_log("node_rrf")
def node_rrf(state:QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    #1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #2. 从状态中获取数据并进行校验
    embedding_chunks, hyde_embedding_chunks = step_1_validate_and_get_data(state)
    #3. 将两路查询结果封装为一个元组，放到大的列表中
    # [(0.6,embedding_chunks),(0.4,hyde_embedding_chunks)]
    # embedding_chunks|hyde_embedding_chunks   [{"chunk_id":xx,"content":xx,"file_tile":...},{},{}]
    data_list = [
        (0.5, embedding_chunks),
        (0.5, hyde_embedding_chunks)
    ]
    #4. 自己实现rrf + 权重融合排序
    rrf_chunks = step_2_use_rrf_rank(data_list)
    state["rrf_chunks"] = rrf_chunks
    #5. 添加节点到已完成列表
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state
@step_log('step_1_validate_and_get_data')
def step_1_validate_and_get_data(state:QueryGraphState):
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 2. 非空校验
    if (not embedding_chunks) or (not hyde_embedding_chunks):
        logger.error(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
        raise ValueError(f"embedding_chunks或者hyde_embedding_chunks为空,业务无法继续,提前终止!")
    # 3. 返回结果
    return embedding_chunks, hyde_embedding_chunks

@step_log('step_2_use_rrf_rank')
def step_2_use_rrf_rank(data_list:list[tuple[str,list]],k:int=60):
    rrf_chunks = []
    score_dict = {}
    chunk_dict = {}
    for weight,chunks  in data_list:
        for index, chunk in enumerate(chunks, start=1):
            score = weight * (1/(k + index))
            chunk_id = chunk.get('chunk_id')
            score_dict[chunk_id] = score + score_dict.get(chunk_id, 0.0)
            chunk_dict[chunk_id] = chunk

    for chunk_id, chunk in chunk_dict.items():
        chunk['score'] = score_dict[chunk_id]
        rrf_chunks.append(chunk)

    logger.debug(f"排序之前,长度:{len(rrf_chunks)},内容: {rrf_chunks}")
    rrf_chunks.sort(key=lambda x: x['score'], reverse=True)
    logger.debug(f"排序之后,长度:{len(rrf_chunks)},内容: {rrf_chunks}")

    rrf_chunks = rrf_chunks[:5]

    return rrf_chunks


if __name__ == "__main__":
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
    }

    emb_res = node_search_embedding(mock_state)
    hyde_res = node_search_embedding_hyde(mock_state)
    logger.debug(f"emb_res:{emb_res},hyde_res:{hyde_res}")
    mock_state["embedding_chunks"] = emb_res.get("embedding_chunks") or []
    mock_state["hyde_embedding_chunks"] = hyde_res.get("hyde_embedding_chunks") or []

    result = node_rrf(mock_state)
    print(result)