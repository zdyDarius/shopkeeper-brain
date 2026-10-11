import time
import sys

from common.logging.logger import node_log,logger
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_running_task, add_done_task

@node_log("node_rerank")
def node_rerank(state:QueryGraphState):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    logger.info("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    time.sleep(1)
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state