import time
import sys

from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task
from common.logging.logger import logger, node_log

@node_log("node_rrf")
def node_rrf(state:QueryGraphState):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    logger.info("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    time.sleep(1)
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state