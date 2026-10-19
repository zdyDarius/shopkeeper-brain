import time
import sys

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.logging.logger import node_log, logger, step_log
from processor.query_processor.state import QueryGraphState
from utils.lm.lm_utils import get_llm_client
from utils.lm.reranker_utils import compute_token_number, get_reranker_model
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5

RERANK_MAX_INPUT_TOKENS: int = 512  # reranker模型的上下窗口
RERANK_SUMMARY_CHAR_RATIO: float = 1.3  # 经验值! 算token -> 字符串
RERANK_MIN_SUMMARY_CHARS: int = 50  # 最低压缩的字符数量



@node_log("node_rerank")
def node_rerank(state:QueryGraphState):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    """
       节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
       """
    #  1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #  2. 从状态中获取数据并进行校验
    rrf_chunks, web_search_docs, rewritten_query = step_1_validate_and_get_data(state)
    #  3. 将rrf结果和web搜索结果进行对齐---统一数据结构
    merged_list = step_2_merge_rrf_and_web(rrf_chunks, web_search_docs)
    #  4. 封装问题-答案对  放到列表中
    question_answer_pair: list[list[str]] = step_3_create_question_answer_pair(merged_list, rewritten_query)
    #  5. 使用重排序模型 进行打分并排序
    step_4_list_score_and_rank(merged_list, question_answer_pair)
    #  6. 动态topk---断崖检测
    reranked_docs = step_5_dynamic_topk(merged_list)
    #  7. 更新状态  reranked_docs
    state["reranked_docs"] = reranked_docs
    #  8. 添加节点到已完成列表
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

@step_log('step_1_validate_and_get_data')
def step_1_validate_and_get_data(state:QueryGraphState):
    rrf_chunks = state.get("rrf_chunks")
    web_search_docs = state.get("web_search_docs")
    rewritten_query = state.get("rewritten_query")
    if (not rrf_chunks) or (not web_search_docs) or (not rewritten_query):
        logger.error(f"rrf_chunks,web_search_docs,rewritten_query参数可能为空,业务无法继续,提前终止!")
        raise ValueError(f"rrf_chunks,web_search_docs,rewritten_query参数可能为空,业务无法继续,提前终止!")
    return rrf_chunks, web_search_docs, rewritten_query

@step_log('step_2_merge_rrf_and_web')
def step_2_merge_rrf_and_web(rrf_chunks, web_search_docs):
    merged_list:list[dict] = []
    for chunk in rrf_chunks:

        merged_list.append({
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "score": 0.0,
            "type": chunk.get("type"),
            "url": ""
        })
    for doc in web_search_docs:
        merged_list.append({
            "title": doc.get("title"),
            # 百炼web搜索返回的字段是snippet,兼容可能的content写法,都没有时兜底空串
            "text": doc.get("snippet") or doc.get("content") or "",
            "score": 0.0,
            "type": 'web',
            "url": doc.get("url")
        })
    logger.info(f"完成两路{len(rrf_chunks)}:{len(web_search_docs)}数据融合,融合后的数量:{len(merged_list)}")
    return merged_list
def step_3_create_question_answer_pair(merged_list, rewritten_query):

    '''拼接问题 + 答案
    :param merged_list:       合并之后的数据  list[{"title":xx,"text":xx,"score":xx,"type":xx,"url":xx,},"title":xx,"text":xx,"score":xx,"type":xx,"url":xx,}...]
    :param rewritten_query:   提问的问题
    :return:    list[list[str]]     例如: [[问题,答案1],[问题,答案2],[问题,答案2]....]
    我们在进行精排打分的时候，使用的是bge-reranker-large，bge-reranker-large上下文长度是514token
    还需要考虑开始、结束、问题和答案的分隔标记，实际上下文长度   514 - 2 - 4 = 508       <s>问题<s><s>答案<s>
    如果问题 + 答案的长度 大于508token，需要我们进行处理，主要的处理方式如下：
        方式1：截取(默认)
        方式2：压缩(我们使用)       调用大语言模型，让其对我们的问题 + 答案进行压缩
        方式3: 换模型
    关于长度的计算      512单位是token，但是在计算 问题 + 答案的时候   我们一般计算的是字符数，所以就涉及字符和token之间进行转换
        在BGE模型中，经验值  1个字符 = 1.3个token ，但是这个不绝对，具体1个字符是多少个token，取决于模型的分词器
        我们可以使用bge-reranker-large提供的一个tokenizer计算字符串对应的token数量'''
    question_answer_pair: list[list[str]] = []
    question_token_len = compute_token_number(rewritten_query)

    for chunk in merged_list:
        # 注意: get的默认值只在键不存在时生效,键存在但值为None时要靠 or 兜底,否则tokenizer.encode(None)会报错
        answer = chunk.get("text") or ''
        answer_token_len = compute_token_number(answer)
        logger.debug(f'{answer_token_len },{ question_token_len}')
        if answer_token_len + question_token_len > (RERANK_MAX_INPUT_TOKENS - 4):
            char_length = max(int((RERANK_MAX_INPUT_TOKENS - 4 - question_token_len) / RERANK_SUMMARY_CHAR_RATIO),
                              RERANK_MIN_SUMMARY_CHARS)

            llm_client = get_llm_client()

            prompt_text = load_prompt('rerank_text_refine',question=rewritten_query,answer=answer,limit=char_length)
            message = HumanMessage(content = prompt_text)
            chains = llm_client | StrOutputParser()
            logger.debug(f"已经触发压缩流程,压缩前{len(answer)}:{answer}")
            answer = chains.invoke([message])
            logger.debug(f"已经触发压缩流程,压缩后{len(answer)}:{answer}")

        question_answer_pair.append([rewritten_query,answer])

    return question_answer_pair





def step_4_list_score_and_rank(merged_list, question_answer_pair):
    # 调用rerank模型进行打分
    rerank_model = get_reranker_model()
    score_list = rerank_model.compute_score(question_answer_pair, normalize=True)
    print(score_list)
    for score, chunk in zip(score_list, merged_list):
        chunk["score"] = score
    logger.debug(f"未排序之前的合并列表:{merged_list}")
    merged_list.sort(key=lambda c: c.get('score', 0.0), reverse=True)
    logger.debug(f"未排序之后的合并列表:{merged_list}")


def step_5_dynamic_topk(merged_list):
    max_top_k = min(RERANK_MAX_TOPK, len(merged_list))

    min_top_k = RERANK_MIN_TOPK
    top_k = max_top_k

    if max_top_k > min_top_k:
        for index in range(min_top_k-1, max_top_k-1):
            current = merged_list[index]
            next_item = merged_list[index+1]

            abs_v = current.get("score") - next_item.get("score")
            radio_v = abs_v/current.get("score")

            if (abs_v > RERANK_GAP_ABS) or (radio_v > RERANK_GAP_RATIO):
                top_k = index + 1
                logger.debug(f"下标:{index}位置发生了断崖!当前值:{current.get('score')},下一个值:{next_item.get('score', 0.0)}")
                break

    reranked_docs = merged_list[:top_k]
    return  reranked_docs



if __name__ == "__main__":
    mock_rrf_chunks = [
        {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍"},
        {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍"},
        {"chunk_id": "local_3", "content": "路漫漫其修远兮，吾将上下而求索路漫漫其修远兮", "title": "励志诗句"},
        {"chunk_id": "local_4", "content": "尚硅谷-让天下没有难学的技术", "title": "广告"}
    ]
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
    ]
    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False,
    }
    result = node_rerank(mock_state)
    print(result)