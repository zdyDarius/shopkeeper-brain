import time
import sys

from common.logging.logger import node_log,logger
from processor.query_processor.state import QueryGraphState
from utils.task_utils import  add_done_task,add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state:QueryGraphState):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    logger.info("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索假设性答案
    print("搜索假设性答案！！")
    time.sleep(1)

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    logger.info("---HyDE 处理结束---")
    return {"hyde_embedding_chunks":"hyde_embedding_chunks"}
