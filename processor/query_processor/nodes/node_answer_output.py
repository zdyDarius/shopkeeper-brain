import time
import sys
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.sse_utils import push_to_session, SSEEvent
from utils.task_utils import add_running_task, add_done_task

@node_log("node_answer_output")
def node_answer_output(state:QueryGraphState):
    """
    节点功能：进行过处理可以是流式输出可以整体输出！
    """
    logger.info("---node_answer_output 节点处理开始---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    session_id = state["session_id"]
    is_stream = state.get("is_stream", True)
    base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。"
    final_text = ""

    if is_stream:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.03)

        image_urls = ["https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png", "https://atguigu.com/images/index_new/logo.png"]

        logger.info(f"流式输出完成，总长度: {len(final_text)}")
    else:
        final_text = base_answer

    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---node_answer_output 节点处理结束---")
    return {"answer": final_text,"image_urls": image_urls}