"""
BGE Reranker 重排序 HTTP 服务（FastAPI）

======================================================================
一、功能说明
======================================================================
把本地 bge-reranker-large 模型发布为 HTTP 服务，对「查询 + 候选文档」
逐对计算相关性分数，返回结构与 utils/lm/reranker_utils.py 的使用方式保持一致：

    现有项目：rerank_model.compute_score([[query, doc], ...], normalize=True)
    本服务：  POST /v1/rerank  ->  {"scores": [0.87, 0.12]}

分数经过 Sigmoid 归一化到 (0, 1)，越大表示相关性越高；
scores 与请求 documents 的顺序严格一一对应，客户端可直接 zip 后排序。

======================================================================
二、依赖与启动
======================================================================
依赖安装（GPU 服务器上按 CUDA 版本安装 torch 后）：
    pip install fastapi uvicorn FlagEmbedding loguru

启动方式（推荐单进程，避免多个 worker 各加载一份模型占满显存）：
    python deploy/bge_reranker_server.py
    # 或者
    uvicorn deploy.bge_reranker_server:app --host 127.0.0.1 --port 8102 --workers 1

环境变量：
    BGE_RERANKER_LARGE       模型目录，默认 /usr-data/models/BAAI/bge-reranker-large
    BGE_RERANKER_DEVICE      推理设备，默认有 CUDA 用 cuda，否则 cpu（如 cuda:0）
    BGE_RERANKER_FP16        是否半精度，默认 GPU 上开启、CPU 上关闭（1/0）
    BGE_RERANKER_BATCH_SIZE  单批打分条数，默认 32（显存不足可调小）
    BGE_RERANKER_MAX_LENGTH  单条样本最大 token 数，默认 512（模型上限）
    BGE_RERANKER_MAX_ITEMS   单次请求最大文档条数，默认 256
    BGE_RERANKER_HOST        监听地址，默认 127.0.0.1
    BGE_RERANKER_PORT        监听端口，默认 8102

======================================================================
三、接口示例
======================================================================
请求：
    curl http://127.0.0.1:8102/v1/rerank \
      -H "Content-Type: application/json" \
      -d '{
        "query": "RS12 怎么测交流电压？",
        "documents": [
          "交流电压测量需要将旋钮置于 V AC，并把表笔接入对应端口。",
          "电阻测量前需要断电并释放电容。"
        ]
      }'

返回：
    {"scores": [0.87, 0.12]}

说明：
    - scores[i] 对应 documents[i]，服务端不改变文档顺序；
    - 模型窗口为 512 token，超长文档会在服务端按模型规则截断。
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field, field_validator

# 日志工具：优先使用项目统一的 loguru，缺失时回退到标准库，保证服务可独立部署
try:
    from loguru import logger
except ImportError:  # pragma: no cover - 仅在未安装 loguru 的部署环境生效
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("bge-reranker-server")


# ============================== 一、服务配置 ==============================
# 模型目录：与现有项目 BGE_RERANKER_LARGE 命名保持一致
MODEL_PATH = os.getenv("BGE_RERANKER_LARGE", "/usr-data/models/BAAI/bge-reranker-large")
# 单批打分条数：GPU 显存充足可调大，显存不足可调小
BATCH_SIZE = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "32"))
# 单条样本最大 token 数：bge-reranker-large 的窗口上限为 512
MAX_LENGTH = int(os.getenv("BGE_RERANKER_MAX_LENGTH", "512"))
# 单次 HTTP 请求允许的最大文档条数，防止超大请求打爆显存
MAX_ITEMS = int(os.getenv("BGE_RERANKER_MAX_ITEMS", "256"))
# 服务监听地址与端口（与 curl 示例中的 8102 保持一致）
HOST = os.getenv("BGE_RERANKER_HOST", "127.0.0.1")
PORT = int(os.getenv("BGE_RERANKER_PORT", "8102"))


def _resolve_device() -> str:
    """
    解析推理设备。

    优先读取环境变量 BGE_RERANKER_DEVICE；未显式配置时自动判断：
    存在可用 CUDA 显卡则使用 "cuda"（即 0 号卡），否则回退到 "cpu"。

    :return: FlagEmbedding 可识别的设备字符串，如 "cuda"、"cuda:0"、"cpu"
    """
    device = os.getenv("BGE_RERANKER_DEVICE", "").strip()
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # torch 不可用时按 CPU 处理，由模型加载阶段给出真实报错
        pass
    return "cpu"


# 推理设备：模块加载时确定，健康检查接口会原样返回，便于部署排查
DEVICE = _resolve_device()
# 半精度开关：GPU 上默认开启可显著降低显存占用；CPU 不支持 fp16，默认关闭
USE_FP16 = os.getenv("BGE_RERANKER_FP16", "1" if DEVICE.startswith("cuda") else "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 进程内模型单例；加载完成后同时挂到 app.state 上供接口使用
_model: FlagReranker | None = None
# 模型加载锁：避免并发请求触发重复加载，造成显存成倍占用
_model_load_lock = threading.Lock()
# 推理锁：同一时刻只允许一个请求进入前向计算，避免并发时显存峰值叠加
_inference_lock = threading.Lock()


# ============================== 二、模型加载与推理 ==============================
def load_model() -> FlagReranker:
    """
    加载 BGE Reranker 模型（进程内单例，线程安全）。

    首次调用时执行：
        1. 校验本地模型目录是否存在，不存在直接报错，避免服务"假启动"；
        2. 构造 FlagReranker，并显式指定单张推理设备，
           避免多卡机器上默认使用全部 GPU 或触发多进程推理；
        3. 缓存实例，后续调用直接复用，避免重复加载耗时和占用显存。

    :return: 初始化完成的 FlagReranker 实例
    :raise RuntimeError: 模型目录不存在时抛出
    :raise Exception: 模型初始化失败时由 FlagEmbedding 向上抛出
    """
    global _model

    # 快速路径：模型已加载，直接复用
    if _model is not None:
        return _model

    with _model_load_lock:
        # 双重检查：等待锁期间可能已被其他线程加载完成
        if _model is not None:
            return _model

        # 部署环境必须使用本地模型目录，避免服务启动时隐式联网下载
        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(
                f"模型目录不存在：{MODEL_PATH}，请检查环境变量 BGE_RERANKER_LARGE 是否指向正确的本地模型目录"
            )

        start = time.perf_counter()
        logger.info(
            f"开始加载 BGE Reranker 模型：path={MODEL_PATH}, device={DEVICE}, "
            f"fp16={USE_FP16}, batch_size={BATCH_SIZE}, max_length={MAX_LENGTH}"
        )
        model = FlagReranker(
            model_name_or_path=MODEL_PATH,  # 本地模型目录，不做联网下载
            devices=DEVICE,                 # 显式指定单卡，避免多卡全占
            use_fp16=USE_FP16,              # 半精度推理，降低显存占用
            batch_size=BATCH_SIZE,          # 内部批量打分条数
            max_length=MAX_LENGTH,          # 单条样本最大 token 数
            normalize=True,                 # 分数做 Sigmoid 归一化，输出 0~1
        )
        _model = model
        cost = time.perf_counter() - start
        logger.info(f"BGE Reranker 模型加载完成，耗时 {cost:.1f}s")
        return _model


def rerank_scores(model: FlagReranker, query: str, documents: list[str]) -> list[float]:
    """
    对「查询 + 候选文档」逐对打分。

    处理流程：
        1. 按现有项目习惯，把查询与每篇文档组装成 [query, document] 二元组；
        2. 在推理锁内调用 compute_score，串行化 GPU 前向计算，
           避免并发请求显存叠加；
        3. normalize=True 对 logits 做 Sigmoid，输出 (0, 1) 区间的相关性分数；
        4. 统一转为 Python float，保证 JSON 可序列化。

    :param model: 已加载的 FlagReranker 实例
    :param query: 查询文本（非空，由接口层校验保证）
    :param documents: 候选文档列表（非空，由接口层校验保证）
    :return: 分数列表，与 documents 顺序严格一一对应
    :raise RuntimeError: 模型返回条数与输入文档条数不一致时抛出
    :raise Exception: 模型推理过程中的异常，由接口层统一捕获
    """
    # 组装成 Cross-Encoder 需要的句子对，顺序与 documents 保持一致
    sentence_pairs = [[query, document] for document in documents]

    with _inference_lock:
        # compute_score 返回 list[float]；normalize=True 与现有 reranker_utils 一致
        scores = model.compute_score(sentence_pairs, normalize=True)

    # 兼容单条输入时模型返回标量的情况，统一成列表
    if isinstance(scores, (int, float)):
        scores = [scores]
    scores = [float(score) for score in scores]

    # 防御性校验：模型输出条数必须与输入文档数一致，避免分数与文档错位
    if len(scores) != len(documents):
        raise RuntimeError(
            f"模型输出条数异常：输入 {len(documents)} 篇文档，返回 {len(scores)} 个分数"
        )
    return scores


# ============================== 三、请求 / 响应数据结构 ==============================
class RerankRequest(BaseModel):
    """
    重排序请求体。

    :param query: 查询文本，不能为空
    :param documents: 候选文档列表，1 ~ BGE_RERANKER_MAX_ITEMS 条；
                      列表中不允许出现空字符串或纯空白文档
    """

    query: str = Field(
        ...,
        min_length=1,
        description="查询文本",
        examples=["RS12 怎么测交流电压？"],
    )
    documents: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_ITEMS,
        description=f"候选文档列表，至少 1 条、最多 {MAX_ITEMS} 条",
        examples=[
            [
                "交流电压测量需要将旋钮置于 V AC，并把表笔接入对应端口。",
                "电阻测量前需要断电并释放电容。",
            ]
        ],
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        """
        校验查询文本：拒绝纯空白内容，并去除首尾空白。

        :param query: 原始查询文本
        :return: 清洗后的查询文本
        :raise ValueError: 查询为空时抛出，FastAPI 会自动转换为 422 响应
        """
        # Pydantic 已保证是字符串，这里只处理纯空白的情况
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query 不能为空或纯空白")
        return cleaned

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, documents: list[str]) -> list[str]:
        """
        校验文档列表：拒绝空字符串 / 纯空白文档，并去除每条文档首尾空白。

        :param documents: 原始文档列表
        :return: 清洗后的文档列表（顺序不变）
        :raise ValueError: 存在空文档时抛出，FastAPI 会自动转换为 422 响应
        """
        cleaned: list[str] = []
        for index, document in enumerate(documents):
            if not isinstance(document, str) or not document.strip():
                raise ValueError(f"第 {index} 篇文档为空或不是字符串")
            cleaned.append(document.strip())
        return cleaned


class RerankResponse(BaseModel):
    """
    重排序响应体。

    :param scores: 相关性分数列表，经 Sigmoid 归一化到 (0, 1)，
                   与请求 documents 顺序严格一一对应
    """

    scores: list[float] = Field(
        description="相关性分数，与 documents 顺序一一对应，值越大相关性越高"
    )


class HealthResponse(BaseModel):
    """
    健康检查响应体。

    :param status: 服务状态，ok=模型可用，loading=模型尚未加载完成
    :param model_loaded: 模型是否已加载
    :param model_path: 模型目录
    :param device: 推理设备
    :param use_fp16: 是否使用半精度
    :param batch_size: 单批打分条数
    :param max_length: 单条样本最大 token 数
    """

    status: str = Field(description="服务状态：ok=模型可用，loading=模型未加载完成")
    model_loaded: bool = Field(description="模型是否已加载到显存/内存")
    model_path: str = Field(description="模型目录")
    device: str = Field(description="推理设备")
    use_fp16: bool = Field(description="是否使用半精度")
    batch_size: int = Field(description="单批打分条数")
    max_length: int = Field(description="单条样本最大 token 数")


# ============================== 四、生命周期管理 ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    启动阶段同步加载模型：模型加载失败时让服务直接启动失败，
    避免出现"服务看起来启动成功、实际所有请求都报错"的情况。
    关闭阶段释放模型引用并清空 CUDA 缓存，便于优雅停机。

    :param app: FastAPI 应用实例
    """
    global _model

    logger.info("服务启动：开始加载 BGE Reranker 模型")
    model = load_model()
    # 挂到 app.state 上供接口层读取，避免接口直接依赖模块全局变量
    app.state.model = model
    logger.info(f"服务已就绪：http://{HOST}:{PORT}/v1/rerank")
    yield

    logger.info("服务关闭：释放模型资源")
    app.state.model = None
    _model = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # 释放缓存失败不影响进程退出
        pass


# 创建 FastAPI 应用，接口文档默认在 /docs
app = FastAPI(
    title="BGE Reranker 重排序服务",
    description="基于 FastAPI 的 bge-reranker-large 重排序推理服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================== 五、HTTP 接口 ==============================
@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health_check() -> HealthResponse:
    """
    健康检查接口。

    返回服务与模型的加载状态、推理设备等基础信息，
    可用于负载均衡探活、部署后的连通性验证。

    :return: 健康检查响应体
    """
    model_loaded = getattr(app.state, "model", None) is not None
    return HealthResponse(
        status="ok" if model_loaded else "loading",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        device=DEVICE,
        use_fp16=USE_FP16,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )


@app.post("/v1/rerank", response_model=RerankResponse, summary="查询-文档相关性打分")
async def create_rerank(request: RerankRequest) -> RerankResponse:
    """
    重排序打分接口。

    对 query 与 documents 中的每篇文档两两打分，返回与 documents
    顺序一致的分数列表；分数越大表示该文档与查询越相关。
    计算逻辑与 utils/lm/reranker_utils.py 中
    compute_score(pairs, normalize=True) 的行为保持一致。

    :param request: 重排序请求体
    :return: 包含 scores 的重排序响应
    :raise HTTPException: 503=模型未就绪，500=推理失败
    """
    model = getattr(app.state, "model", None)
    if model is None:
        # 正常启动流程下不会出现，兜底处理模型未加载的情况
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    documents = request.documents
    logger.info(f"收到重排序请求：1 条查询，{len(documents)} 篇文档")
    start = time.perf_counter()

    try:
        # 模型打分是同步的 CPU/GPU 阻塞操作，放入线程池执行，避免阻塞事件循环
        scores = await run_in_threadpool(rerank_scores, model, request.query, documents)
    except Exception as exc:
        logger.exception("重排序失败")
        raise HTTPException(status_code=500, detail=f"重排序失败：{exc}") from exc

    cost_ms = (time.perf_counter() - start) * 1000
    logger.info(f"重排序完成：{len(documents)} 篇文档，耗时 {cost_ms:.1f}ms")
    return RerankResponse(scores=scores)


# ============================== 六、启动入口 ==============================
if __name__ == "__main__":
    """
    直接运行入口。

    注意：workers 固定为 1——每个 worker 都会独立加载一份模型，
    多进程会成倍占用显存，模型服务应由单进程 + 线程池并发来支撑。
    """
    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        workers=1,
        log_level="info",
    )
