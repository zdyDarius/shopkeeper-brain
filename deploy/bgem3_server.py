"""
BGE-M3 向量化 HTTP 服务（FastAPI）

======================================================================
一、功能说明
======================================================================
把本地 BGE-M3 模型发布为 HTTP 服务，一次请求同时返回「稠密向量（dense）」
与「稀疏向量（sparse）」，返回结构与 utils/lm/embedding_utils.py 中的
generate_embeddings() 完全一致（dense 为嵌套列表，sparse 为字典列表），
现有项目只需把本地函数调用换成 HTTP 请求即可完成对接。

======================================================================
二、依赖与启动
======================================================================
依赖安装（GPU 服务器上按 CUDA 版本安装 torch 后）：
    pip install fastapi uvicorn "pymilvus[model]" loguru

启动方式（推荐单进程，避免多个 worker 各加载一份模型占满显存）：
    python deploy/bgem3_server.py
    # 或者
    uvicorn deploy.bgem3_server:app --host 127.0.0.1 --port 6006 --workers 1

环境变量：
    BGE_M3_PATH     模型目录，默认 /usr-data/models/BAAI/bge-m3
    BGE_DEVICE      推理设备，默认有 CUDA 用 cuda，否则 cpu（如 cuda:0）
    BGE_FP16        是否半精度，默认 GPU 上开启、CPU 上关闭（1/0）
    BGE_BATCH_SIZE  单批编码条数，默认 32（显存不足可调小）
    BGE_MAX_ITEMS   单次请求最大文本条数，默认 256
    BGE_HOST        监听地址，默认 127.0.0.1
    BGE_PORT        监听端口，默认 6006

======================================================================
三、接口示例
======================================================================
请求：
    curl http://127.0.0.1:6006/embeddings \
      -H "Content-Type: application/json" \
      -d '{"embedding_documents":["RS12 万用表如何测量交流电压？","如何判断电阻量程？"]}'

返回：
    {
      "dense": [[...], [...]],
      "sparse": [{"12345": 0.28, "67890": 0.51}, {...}]
    }

注意：
    1. JSON 规范要求对象 key 必须是字符串，因此稀疏向量的 token id 在响应中
       是字符串；写入 Milvus 前请转回 int：
       {int(k): float(v) for k, v in sparse_item.items()}
    2. 响应中的 dense / sparse 均与请求文本顺序一一对应。
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pydantic import BaseModel, Field, field_validator

# 日志工具：优先使用项目统一的 loguru，缺失时回退到标准库，保证服务可独立部署
try:
    from loguru import logger
except ImportError:  # pragma: no cover - 仅在未安装 loguru 的部署环境生效
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("bgem3-server")


# ============================== 一、服务配置 ==============================
# 模型目录：与现有项目 BGE_M3_PATH 命名保持一致，默认使用云 GPU 服务器上的固定路径
MODEL_PATH = os.getenv("BGE_M3_PATH", "/usr-data/models/BAAI/bge-m3")
# 单批编码条数：GPU 显存充足可调大，显存不足可调小
BATCH_SIZE = int(os.getenv("BGE_BATCH_SIZE", "32"))
# 单次 HTTP 请求允许的最大文本条数，防止超大请求打爆显存
MAX_ITEMS = int(os.getenv("BGE_MAX_ITEMS", "256"))
# 服务监听地址与端口（与 curl 示例中的 6006 保持一致）
HOST = os.getenv("BGE_HOST", "0.0.0.0")
PORT = int(os.getenv("BGE_PORT", "6003"))


def _resolve_device() -> str:
    """
    解析推理设备。

    优先读取环境变量 BGE_DEVICE；未显式配置时自动判断：
    存在可用 CUDA 显卡则使用 "cuda"，否则回退到 "cpu"。

    :return: FlagEmbedding 可识别的设备字符串，如 "cuda"、"cuda:0"、"cpu"
    """
    device = os.getenv("BGE_DEVICE", "").strip()
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
USE_FP16 = os.getenv("BGE_FP16", "1" if DEVICE.startswith("cuda") else "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# 进程内模型单例；加载完成后同时挂到 app.state 上供接口使用
_model: BGEM3EmbeddingFunction | None = None
# 模型加载锁：避免并发请求触发重复加载，造成显存成倍占用
_model_load_lock = threading.Lock()
# 推理锁：同一时刻只允许一个请求进入前向计算，避免并发时显存峰值叠加
_inference_lock = threading.Lock()


# ============================== 二、模型加载与推理 ==============================
def load_model() -> BGEM3EmbeddingFunction:
    """
    加载 BGE-M3 模型（进程内单例，线程安全）。

    首次调用时执行：
        1. 校验本地模型目录是否存在，不存在直接报错，避免服务"假启动"；
        2. 构造 BGEM3EmbeddingFunction，开启 normalize_embeddings，
           使稠密向量完成 L2 归一化，与现有 embedding_utils 的行为保持一致；
        3. 缓存实例，后续调用直接复用，避免重复加载耗时和占用显存。

    :return: 初始化完成的 BGEM3EmbeddingFunction 实例
    :raise RuntimeError: 模型目录不存在时抛出
    :raise Exception: 模型初始化失败时由 pymilvus/FlagEmbedding 向上抛出
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
                f"模型目录不存在：{MODEL_PATH}，请检查环境变量 BGE_M3_PATH 是否指向正确的本地模型目录"
            )

        start = time.perf_counter()
        logger.info(
            f"开始加载 BGE-M3 模型：path={MODEL_PATH}, device={DEVICE}, "
            f"fp16={USE_FP16}, batch_size={BATCH_SIZE}"
        )
        model = BGEM3EmbeddingFunction(
            model_name=MODEL_PATH,          # 本地模型目录，不做联网下载
            device=DEVICE,                  # 推理设备
            use_fp16=USE_FP16,              # 半精度推理，降低显存占用
            normalize_embeddings=True,      # 稠密向量 L2 归一化，适配 Milvus IP 检索
            batch_size=BATCH_SIZE,          # 内部批量编码条数
        )
        _model = model
        cost = time.perf_counter() - start
        logger.info(f"BGE-M3 模型加载完成，耗时 {cost:.1f}s")
        return _model


def encode_texts(
    model: BGEM3EmbeddingFunction,
    texts: list[str],
    mode: Literal["document", "query"] = "document",
) -> dict:
    """
    生成稠密 + 稀疏混合向量，并转换为可直接 JSON 序列化的结构。

    处理流程：
        1. 在推理锁内调用模型，串行化 GPU 前向计算，避免并发请求显存叠加；
        2. 稠密向量：逐条转成 Python list（np.ndarray / list 两种返回形式都兼容）；
        3. 稀疏向量：BGEM3EmbeddingFunction 返回 CSR 矩阵，按 indptr 逐行切片，
           还原为 {token_id: weight} 字典；
        4. 所有键值统一转成 Python 原生类型（str/int/float），满足 JSON 序列化
           与 Milvus 入库要求，彻底规避 numpy 类型不可序列化的问题。

    :param model: 已加载的 BGEM3EmbeddingFunction 实例
    :param texts: 待向量化的文本列表（非空，由接口层校验保证）
    :param mode: 编码模式，document=按文档编码（默认），query=按查询编码
    :return: {"dense": [[float, ...], ...], "sparse": [{str: float, ...}, ...]}
             两个列表均与输入文本顺序一一对应
    :raise Exception: 模型编码过程中的异常，由接口层统一捕获并转换为 HTTP 错误
    """
    with _inference_lock:
        # BGE-M3 的 query / document 编码接口分开提供，便于后续接入带指令的模型
        if mode == "query":
            embeddings = model.encode_queries(texts)
        else:
            embeddings = model.encode_documents(texts)

        # 稠密向量：兼容 numpy 数组与 list 两种返回形式，统一转为 Python float 列表
        dense = [
            np.asarray(vector, dtype=np.float32).tolist()
            for vector in embeddings["dense"]
        ]

        # 稀疏向量：模型返回 scipy CSR 矩阵，通过 indptr 取出第 i 行非零元素
        csr = embeddings["sparse"]
        sparse = []
        for i in range(len(texts)):
            start, end = int(csr.indptr[i]), int(csr.indptr[i + 1])
            # tolist() 将 np.int64 / np.float32 转为 Python 原生 int / float
            indices = csr.indices[start:end].tolist()
            values = csr.data[start:end].tolist()
            # JSON 对象 key 必须为字符串，统一用 str() 转换；客户端入库前再转回 int
            sparse.append({str(idx): float(weight) for idx, weight in zip(indices, values)})

        # 防御性校验：模型输出条数必须与输入一致，避免返回错位数据
        if len(dense) != len(texts) or len(sparse) != len(texts):
            raise RuntimeError(
                f"模型输出条数异常：输入 {len(texts)} 条，dense {len(dense)} 条，sparse {len(sparse)} 条"
            )

    return {"dense": dense, "sparse": sparse}


# ============================== 三、请求 / 响应数据结构 ==============================
class EmbeddingRequest(BaseModel):
    """
    向量化请求体。

    :param embedding_documents: 待向量化的文本列表，1 ~ BGE_MAX_ITEMS 条；
                                列表中不允许出现空字符串或纯空白字符串
    :param mode: 编码模式，document=文档（默认），query=查询
    """

    embedding_documents: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_ITEMS,
        description=f"待向量化的文本列表，至少 1 条、最多 {MAX_ITEMS} 条",
        examples=[["RS12 万用表如何测量交流电压？", "如何判断电阻量程？"]],
    )
    mode: Literal["document", "query"] = Field(
        default="document",
        description="编码模式：document=文档（默认），query=查询",
    )

    @field_validator("embedding_documents")
    @classmethod
    def validate_documents(cls, documents: list[str]) -> list[str]:
        """
        校验文本列表：拒绝空字符串 / 纯空白文本，并去除每条文本首尾空白。

        :param documents: 原始文本列表
        :return: 清洗后的文本列表
        :raise ValueError: 存在空文本时抛出，FastAPI 会自动转换为 422 响应
        """
        cleaned: list[str] = []
        for index, text in enumerate(documents):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {index} 条文本为空或不是字符串")
            cleaned.append(text.strip())
        return cleaned


class EmbeddingResponse(BaseModel):
    """
    向量化响应体。

    :param dense: 稠密向量列表，与请求文本顺序一一对应，每项为固定维度的 float 列表
    :param sparse: 稀疏向量列表，与请求文本顺序一一对应，每项形如 {"token_id": 权重}
    """

    dense: list[list[float]] = Field(description="稠密向量，与输入文本顺序一一对应")
    sparse: list[dict[str, float]] = Field(
        description="稀疏向量（token_id -> 权重），与输入文本顺序一一对应"
    )


class HealthResponse(BaseModel):
    """
    健康检查响应体。

    :param status: 服务状态，ok=模型可用，loading=模型尚未加载完成
    :param model_loaded: 模型是否已加载
    :param model_path: 模型目录
    :param device: 推理设备
    :param use_fp16: 是否使用半精度
    :param batch_size: 单批编码条数
    """

    status: str = Field(description="服务状态：ok=模型可用，loading=模型未加载完成")
    model_loaded: bool = Field(description="模型是否已加载到显存/内存")
    model_path: str = Field(description="模型目录")
    device: str = Field(description="推理设备")
    use_fp16: bool = Field(description="是否使用半精度")
    batch_size: int = Field(description="单批编码条数")


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

    logger.info("服务启动：开始加载 BGE-M3 模型")
    model = load_model()
    # 挂到 app.state 上供接口层读取，避免接口直接依赖模块全局变量
    app.state.model = model

    try:
        # 输出维度信息，方便部署时确认模型加载正确
        dims = model.dim
        logger.info(
            f"模型信息：dense 维度={dims['dense']}，稀疏词表大小={dims['sparse']}"
        )
    except Exception:
        # 不同版本 pymilvus 的 dim 属性可能存在差异，仅影响日志展示，不影响服务
        logger.debug("获取模型维度信息失败，跳过日志展示")

    logger.info(f"服务已就绪：http://{HOST}:{PORT}/embeddings")
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
    title="BGE-M3 向量化服务",
    description="基于 FastAPI 的 BGE-M3 混合向量（dense + sparse）推理服务",
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
    )


@app.post("/embeddings", response_model=EmbeddingResponse, summary="生成稠密+稀疏混合向量")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    混合向量生成接口。

    请求体中的 embedding_documents 为待向量化文本列表；
    响应中的 dense / sparse 与输入文本顺序一一对应，数据结构与
    utils/lm/embedding_utils.py::generate_embeddings 保持一致。

    :param request: 向量化请求体
    :return: 包含 dense 与 sparse 的向量化响应
    :raise HTTPException: 503=模型未就绪，500=推理失败
    """
    model = getattr(app.state, "model", None)
    if model is None:
        # 正常启动流程下不会出现，兜底处理模型未加载的情况
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试")

    texts = request.embedding_documents
    logger.info(f"收到向量化请求：{len(texts)} 条文本，mode={request.mode}")
    start = time.perf_counter()

    try:
        # 模型编码是同步的 CPU/GPU 阻塞操作，放入线程池执行，避免阻塞事件循环
        result = await run_in_threadpool(encode_texts, model, texts, request.mode)
    except Exception as exc:
        logger.exception("向量生成失败")
        raise HTTPException(status_code=500, detail=f"向量生成失败：{exc}") from exc

    cost_ms = (time.perf_counter() - start) * 1000
    logger.info(f"向量化完成：{len(texts)} 条文本，耗时 {cost_ms:.1f}ms")
    return EmbeddingResponse(**result)


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
