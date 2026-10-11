from datetime import datetime
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, FileResponse

from common.logging.logger import logger
from processor.import_processor.main_graph import kb_import_app
from processor.import_processor.state import create_default_state, ImportGraphState
from utils.path_util import PROJECT_ROOT
from utils.task_utils import get_done_task_list, get_running_task_list, get_task_status, update_task_status, \
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED
import logging

app = FastAPI()
UPLOAD_DIR = PROJECT_ROOT / "output" / "upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 在服务器端配置跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class HealthCheckFilter(logging.Filter):
    """过滤掉 /health 健康检查的访问日志,避免前端轮询刷屏控制台"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 访问日志格式: 127.0.0.1:xxxx - "GET /health HTTP/1.1" 200
        return '"GET /health' not in record.getMessage()
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

@app.get("/")
def import_page():
    import_page_path = PROJECT_ROOT / "api·····" / "import.html"

    return FileResponse(import_page_path, media_type="text/html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/task/{task_id}")
async def get_status(task_id: str):
    done_task_list = get_done_task_list(task_id)
    running_task_list = get_running_task_list(task_id)
    task_status = get_task_status(task_id)
    return {
        "code": 200,
        "task_id": task_id,
        "done_list": done_task_list,
        "running_list": running_task_list,
        "status": task_status
    }
@app.post("/upload")
async def upload(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        # 1. 校验文件类型(导入图只支持 pdf / md)
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".pdf", ".md"}:
            raise HTTPException(status_code=400, detail=f"仅支持pdf/md文件,当前文件类型:{suffix}")
        task_id = uuid.uuid4().hex

        today_str = datetime.now().strftime("%Y%m%d")
        task_local_dir: Path = UPLOAD_DIR / today_str/task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        file_path_obj: Path =  task_local_dir / f"{Path(file.filename).name}"
        data = await file.read()
        file_path_obj.write_bytes(data)
        # 3. 构造图状态(参照 01_test_import_main_graph.py)
        state = create_default_state(
            task_id=task_id,
            local_file_path=str(file_path_obj),
            local_dir=str(task_local_dir),  # 中间文件输出目录
            is_pdf_read_enabled=(suffix == ".pdf"),
            is_md_read_enabled=(suffix == ".md"),
        )

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


def run_import_task(state: ImportGraphState):
    """
    后台执行的文档导入任务(参照 01_test_import_main_graph.py 的执行方式)
    注意:必须使用同步def函数,starlette会把它放到线程池中执行,不会阻塞事件循环
    """
    task_id = state["task_id"]
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        logger.info(f"[{task_id}] 后台导入任务开始执行,文件:{state['local_file_path']}")
        final_state = {}
        # 流式执行LangGraph全流程,打印节点执行进度
        for step in kb_import_app.stream(state, stream_mode="updates"):
            for node_name, updates in step.items():
                logger.info(f"[{task_id}] ✅ 节点执行完成:{node_name}")
                final_state.update(updates)
        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"[{task_id}] 后台导入任务执行成功,切片数:{len(final_state.get('chunks', []))}")
    except Exception:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(f"[{task_id}] 后台导入任务执行失败")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)