from datetime import datetime
import json
from pathlib import Path
from pydantic import BaseModel, Field

import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException,Request

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse, StreamingResponse

from common.logging.logger import logger
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.path_util import PROJECT_ROOT
from utils.sse_utils import sse_generator, push_to_session, SSEEvent, create_sse_queue
from utils.task_utils import get_done_task_list, get_running_task_list, get_task_status, update_task_status, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, clear_task
import logging

app = FastAPI()


# 在服务器端配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/chat.html")
def chat():
    return FileResponse(PROJECT_ROOT / "api" /'dist'/ "index.html")



@app.get("/stream/{session_id}")
def get_stream(session_id: str, request: Request):
    logger.info(f"{datetime.now()}完成{session_id}对应的流式响应接口!")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
    )


# 定义一个类，用于封装客户端查询的时候传递给服务器端的参数
class QueryRequest(BaseModel):
    query: str = Field(..., description="查询内容")

    session_id: str = Field(None, description="会话ID")

    is_stream: bool = Field(False, description="是否流式返回")
def get_query_graph(session_id: str,is_stream: bool,user_query: str):
    try:
        # 在执行图节点前，应该先将task_utils中维护的数据清除掉
        clear_task(session_id)
        # 更新全局状态为运行中
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)

        state = create_query_default_state(
            session_id=session_id,
            is_stream=is_stream,
            original_query=user_query
        )
        logger.info(f"查询图开始执行,传入参数:\n{json.dumps(state, indent=4, ensure_ascii=False)}")
        final_state = query_app.invoke(state)
        logger.info(f"查询图执行结束,返回结果:\n{json.dumps(final_state, indent=4, ensure_ascii=False)}")
        # 更新全局状态为已完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
        if is_stream:
            push_to_session(
                session_id,
                SSEEvent.FINAL,
                {
                    "answer": final_state.get("answer"),
                    "status": "completed",
                    "image_urls": final_state.get("image_urls"),
                    'name': 'zdy'
                }
            )
        return final_state


    except Exception as e:
        # 更新全局状态为失败
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        logger.exception(f"{user_query}:执行图流程报错!")
        # 向客户端通过SSE推送事件类型为"error"的消息
        push_to_session(session_id=session_id, event=SSEEvent.ERROR, data={
            "error": str(e)
        })


@app.post("/query")
def query(query_request: QueryRequest, background_tasks: BackgroundTasks):
    session_id = query_request.session_id
    user_query = query_request.query

    is_stream = query_request.is_stream

    if is_stream:
        create_sse_queue(session_id)

        background_tasks.add_task(get_query_graph, session_id, is_stream, user_query)

        logger.info(f"[{session_id}] 流式任务已提交至后台执行")
        return {
            "message": "结果正在处理中...",
            "session_id": session_id
        }
    else:
        final_state = get_query_graph(session_id,is_stream ,user_query)
        done_list = get_done_task_list(session_id)

        return JSONResponse(status_code=200, content={
            "session_id": session_id,
            "message": f"{session_id}已经完成问题解析!",
            "answer": final_state.get("answer"),
            "done_list": done_list,
            "image_urls": final_state.get("image_urls", [])
        })

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)


