import time
import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.milvus_config import milvus_config
from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.clients.milvus_utils import create_hybrid_search_requests, get_milvus_client,hybrid_search
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import  add_done_task,add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state:QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    logger.info("---HyDE 开始处理---")
    #  1. 添加节点到运行时列表
    add_running_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    #  2. 从状态中获取数据并进行校验
    item_names, rewritten_query = step_1_validate_and_get_data(state)
    #  3. 调用模型生成假设性答案
    hyde_answer = step_2_call_llm_by_rewritten_query(rewritten_query)
    #  4. 向量化查询
    real_response: list[dict] = step_3_select_chunks_in_milvus(item_names, rewritten_query, hyde_answer)
    #  5. 对查询结果进行格式化处理
    hyde_embedding_chunks = step_4_after_deal_milvus_result(real_response)
    #  6. 添加节点到已完成列表
    add_done_task(state.get("session_id"), sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": hyde_embedding_chunks}

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state:QueryGraphState):
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if not item_names or not rewritten_query:
        logger.warning(f"item_names或者rewritten_query为空，item_names={item_names}, rewritten_query={rewritten_query}")
        raise ValueError(f"item_names或者rewritten_query为空,业务无法继续,提前终止!")
    return item_names, rewritten_query

@step_log("step_2_validate_and_get_data")
def step_2_call_llm_by_rewritten_query(rewritten_query:QueryGraphState):

    llm = get_llm_client()
    promt_text = load_prompt('hyde_prompt',rewritten_query=rewritten_query)

    message = HumanMessage(content=promt_text)
    chains = llm | StrOutputParser()

    result = chains.invoke([message])
    logger.info(f"基于:{rewritten_query},模型给与的假设性回答:{result}")
    # 返回
    return result
@step_log("step_3_select_chunks_in_milvus")
def step_3_select_chunks_in_milvus(item_names:list, rewritten_query,hyde_answer):

    result = generate_embeddings(["问题:"+rewritten_query+"\n 假设性回答:"+hyde_answer])
    '''
    result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
    '''
    dense_vectors = result["dense"][0]
    sparse_vectors = result["sparse"][0]

    # 封装AnnSearchRequest列表
    """
       构建Milvus混合搜索请求对象
       分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
       :param dense_vector: 文本生成的稠密向量
       :param sparse_vector: 文本生成的稀疏向量
       :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
       :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
       :param expr: 搜索过滤表达式，用于精准筛选数据
       :param limit: 单向量搜索返回结果数量，默认5
       :return: 搜索请求列表，包含[dense_req, sparse_req]
       """
    reqList = create_hybrid_search_requests(dense_vectors, sparse_vectors, expr=f'item_name in {item_names}')
    # 混合检索
    """
       执行Milvus稠密+稀疏向量混合搜索
       基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
       :param client: MilvusClient实例
       :param collection_name: 集合名称
       :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
       :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
       :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
       :param limit: 混合搜索最终返回结果数量，默认5
       :param output_fields: 需要返回的字段列表，默认返回item_name
       :param search_params: 搜索参数，如ef/topk等，默认None
       :return: 混合搜索结果列表，搜索失败返回None
       """
    response = hybrid_search(
        client=get_milvus_client(),
        collection_name=milvus_config.chunks_collection,
        reqs=reqList,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        output_fields=[
            "chunk_id",
            "file_title",
            "parent_title",
            "title",
            "part",
            "item_name",
            "content"
        ]
    )
    return response[0]

@step_log("step_4_after_deal_milvus_result")
def step_4_after_deal_milvus_result(real_response)->list[dict]:
    hyde_embedding_chunks: list[dict] = []

    # 对查询结果进行遍历
    if real_response:
        for item in real_response:
            chunk = item.get("entity")
            chunk["score"] = item.get("distance", 0.0)
            chunk["type"] = "milvus"
            hyde_embedding_chunks.append(chunk)

    logger.info(f"完成了问题向量检索!检索的数量:{len(hyde_embedding_chunks)}")
    return hyde_embedding_chunks
