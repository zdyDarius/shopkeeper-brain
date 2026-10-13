import time
import sys
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState
from utils.clients.mongo_history_utils import save_chat_message
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
    image_urls=[]
    if is_stream:
        for ch in base_answer:
            final_text += ch
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.03)

        image_urls = ["http://localhost:9000/knowledge-base-files/upload-images/03_%E4%BA%91%E5%B2%AD%E4%BA%91%E5%8D%97%E6%97%A5%E6%99%92%E6%8C%82%E8%80%B3%E5%92%96%E5%95%A1/c169fdd638f6b905781ccc96000555b6c47f2145775266f9ed87af1bd6f2c37c.jpg", "https://atguigu.com/images/index_new/logo.png"]

        logger.info(f"流式输出完成，总长度: {len(final_text)}")
    else:
        final_text = base_answer
    save_chat_message(
        session_id=state.get("session_id"),
        role='assistant',
        text=final_text,
        rewritten_query="这是重写的问题",
        item_names=["商品A", "商品B"],
        image_urls=image_urls
    )
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    logger.info("---node_answer_output 节点处理结束---")
    return {"answer": final_text,"image_urls": image_urls}