import time
import sys

from common.logging.logger import node_log,logger
from processor.query_processor.state import QueryGraphState
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

    # 后面会调用大模型，进行逻辑处理
    time.sleep(1)
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    logger.info("---node_item_name_confirm---处理结束")

    return {"item_names": ["示例商品"]}