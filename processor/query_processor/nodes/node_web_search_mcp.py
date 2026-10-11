import time
import sys

from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task,add_running_task
from common.logging.logger import logger, node_log

@node_log("node_web_search_mcp")
def node_web_search_mcp(state:QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    logger.info("---node-web-search-mcp处理---")

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
    time.sleep(1)
    # 调用mcp外部引擎
    logger.info(f"调用外部mcp引擎")

    logger.info("---node-web-search-mcp处理结束---")
    return {"web_search_docs":"web_search_docs"}
