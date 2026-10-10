"""
    fastapi_bk
        开发框架,主要用于开发基于python语言的web服务(接口)
        接收前端请求，并对请求进行处理，然后将结果响应给前端页面
    协程---fastapi支持协程，并且事件循环不需要我们自己维护，由服务器uvicorn维护
    简单的fastapi入门案例
        @app.get("/hello")
        def say_hello():
            return "Hello World!"
    参数传递的形式
        请求参数
            @app.get("/login")
            def login(username: str, password: str,age:int):
                return {"username": username, "password": password, "age": age}
        路径参数
            @app.get("/student/{std_id}")
            def get_student_info(std_id):
                return {"std_id": std_id}
        请求体参数
            请求行
            请求头
            空行
            请求体  ---post、put
            class User(BaseModel):
            username: str
            password: str

            @app.post("/login")
            def login(user:User):
                return {"username": user.username, "password": user.password}
    向客户端响应的类型
        JSONResponse
            @app.get("/hello")
            def say_hello():
                # return {"message": "Hello World!"}
                return JSONResponse(content={"hello": "world"}
        PlainTextResponse
            @app.get("/hello")
            def say_hello():
                # return "Hello World!"
                return PlainTextResponse(content="Hello", status_code=200)
        FileResponse
            @app.get("/download")
            def download():
                pdf_path = "D:dev\workspace\shopkeeper-brain-0708\doc\hak180产品安全手册.pdf"
                # 返回文件并指定下载文件名
                return FileResponse(
                    path=pdf_path,
                    filename="haodongxi.pdf",
                    media_type=guess_type(pdf_path)[0]
                )

        HTMLResponse
            @app.get("/hello")
            def hello(name: str = "游客"):
                html_content = f'''
                <html>
                    <body>
                        <h1>你好，{name}！</h1>
                    </body>
                </html>
                '''
                return HTMLResponse(content=html_content, status_code=200)

        RedirectResponse
            @app.get("/old-path")
            def redirect_old_path():
                # 重定向到 /new-path，状态码 307 表示临时重定向
                return RedirectResponse(url="/new-path", status_code=307)

            @app.get("/new-path")
            def new_path():
                return {"message": "这是新接口"}

        StreamingResponse
            async def generate_stream():
                # 模拟流式输出（逐字返回）
                words = ["你", "好", "，", "这", "是", "流", "式", "响", "应"]
                for word in words:
                    await asyncio.sleep(0.5)
                    yield word.encode("utf-8")  # 流式输出需返回字节流

            @app.get("/stream")
            async def stream_response():
                return StreamingResponse(generate_stream(), media_type="text/event-stream")
    文件上传
        @app.post("/upload")
        async def upload(file: UploadFile):
            try:

                # file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file_path_obj:Path = PROJECT_ROOT / "output" / file.filename

                data = await file.read()
                file_path_obj.write_bytes(data)

                return JSONResponse(
                    status_code=200,
                    content={
                        "code": 0,
                        "msg": "文件上传成功",
                        "data": {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "file_size": f"{file.size} 字节",  # 文件大小
                        }
                    }
                )
            except Exception as e:
                # 异常捕获：返回友好的错误信息
                raise HTTPException(
                    status_code=500,
                    detail=f"文件上传失败：{str(e)}"
                )
"""


import asyncio
import logging
import time
import uuid
from mimetypes import guess_type
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from starlette.responses import JSONResponse, PlainTextResponse, FileResponse, HTMLResponse, RedirectResponse, \
    StreamingResponse

from common.logging.logger import logger
from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.path_util import PROJECT_ROOT
from utils.task_utils import get_done_task_list, get_running_task_list,get_task_status

# 创建fastapi应用实例
app = FastAPI()


class HealthCheckFilter(logging.Filter):
    """过滤掉 /health 健康检查的访问日志,避免前端轮询刷屏控制台"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 访问日志格式: 127.0.0.1:xxxx - "GET /health HTTP/1.1" 200
        return '"GET /health' not in record.getMessage()


# 挂到uvicorn的访问日志器上(uvicorn默认配置不会清除已存在的filter,启动后依然生效)
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

# 上传文件保存目录
UPLOAD_DIR = PROJECT_ROOT / "output" / "upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 任务状态表(内存版,重启丢失;生产环境应换成redis/mysql)
TASK_STATUS: dict[str, str] = {}


# 定义一个模拟的耗时任务
def write_log(message: str):
    time.sleep(5) # 模拟耗时 2 秒
    with open("log.txt", "a") as log:
        log.write(message + "\n")

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    # 1. 添加任务到后台队列
    background_tasks.add_task(write_log, f"Notification sent to {email}")
    # 2. 立即返回响应给用户，不需要等待 write_log 执行完毕
    return {"message": "Notification sent in the background"}


def run_import_task(state: ImportGraphState):
    """
    后台执行的文档导入任务(参照 01_test_import_main_graph.py 的执行方式)
    注意:必须使用同步def函数,starlette会把它放到线程池中执行,不会阻塞事件循环
    """
    task_id = state["task_id"]
    try:
        logger.info(f"[{task_id}] 后台导入任务开始执行,文件:{state['local_file_path']}")
        final_state = {}
        # 流式执行LangGraph全流程,打印节点执行进度
        for step in kb_import_app.stream(state, stream_mode="updates"):
            for node_name, updates in step.items():
                logger.info(f"[{task_id}] ✅ 节点执行完成:{node_name}")
                final_state.update(updates)
        TASK_STATUS[task_id] = "success"
        logger.info(f"[{task_id}] 后台导入任务执行成功,切片数:{len(final_state.get('chunks', []))}")
    except Exception:
        TASK_STATUS[task_id] = "failed"
        logger.exception(f"[{task_id}] 后台导入任务执行失败")


@app.post('/upload')
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        # 1. 校验文件类型(导入图只支持 pdf / md)
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail=f"仅支持pdf/md文件,当前文件类型:{suffix}")

        # 2. 生成任务ID并保存上传文件(文件名加前缀避免重名覆盖)
        # 每次上传前都确保目录存在:模块导入时的 mkdir 只执行一次,
        # 服务启动后 output/ 目录被清理过就会写失败(实测踩过)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        task_id = uuid.uuid4().hex
        file_path_obj: Path = UPLOAD_DIR /f'{task_id}'/f"{Path(file.filename).name}"
        data = await file.read()
        file_path_obj.write_bytes(data)

        # 3. 构造图状态(参照 01_test_import_main_graph.py)
        state = create_default_state(
            task_id=task_id,
            local_file_path=str(file_path_obj),
            local_dir=str(PROJECT_ROOT / "output"),  # 中间文件输出目录
            is_pdf_read_enabled=(suffix == ".pdf"),
            is_md_read_enabled=(suffix == ".md"),
        )

        # 4. 添加任务到后台队列
        TASK_STATUS[task_id] = "processing"
        background_tasks.add_task(run_import_task, state)

        # 5. 立即返回上传结果给前端,不需要等待导入流程执行完毕
        return JSONResponse(
            status_code=200,
            content={
                "code": 0,
                "msg": "文件上传成功,导入任务已进入后台队列",
                "data": {
                    "task_id": task_id,  # 前端凭此ID轮询任务状态
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_size": f"{len(data)} 字节",  # 文件大小
                    "status": "processing",
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        # 异常捕获：返回友好的错误信息
        logger.exception("文件上传失败")
        raise HTTPException(
            status_code=500,
            detail=f"文件上传失败：{str(e)}"
        )


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """前端轮询接口:任务状态 + 节点进度"""
    status = TASK_STATUS.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {
        "code": 0,
        "data": {
            "task_id": task_id,
            "status": status,  # processing/success/failed,保持不变
            "done_list": get_done_task_list(task_id),       # 已完成节点(自动中文)
            "running_list": get_running_task_list(task_id), # 正在进行节点(自动中文)
        },
    }


@app.get("/health")
async def get_health():
    return {"code": 0}



if __name__ == "__main__":
    # 8000端口已被node进程占用,改用8001
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # uvicorn.run(app, host="127.0.0.1", port=8001)