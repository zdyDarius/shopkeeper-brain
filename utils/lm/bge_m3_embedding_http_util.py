"""
BGE-M3 混合向量 HTTP 客户端工具

用于替换本地模型版工具（utils/lm/embedding_utils.py），
把「调用本地 BGEM3EmbeddingFunction」改为「请求远端 BGE-M3 HTTP 服务」。

服务端实现见 deploy/bgem3_server.py，接口约定：
    POST {BGE_M3_URL}
    请求体：{"embedding_documents": ["文本1", "文本2", ...], "mode": "document" | "query"}
    响应体：{"dense": [[...], ...], "sparse": [{"token_id": 权重}, ...]}

调用方式由环境变量 BGE_MODEL_MODE 控制：
    http  = 走远端 HTTP 服务（默认，推荐）
    local = 走本地模型（复用 utils/lm/embedding_utils.py）

本模块对外保持与原 generate_embeddings() 一致的返回结构：
    {"dense": [[float, ...], ...], "sparse": [{int: float, ...}, ...]}
其中 sparse 的 token_id 已从 JSON 的字符串键还原为 int，
可直接写入 Milvus 的稀疏向量字段。
"""

import os
import time

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common.logging.logger import logger

# 加载 .env 配置：保证单独导入本模块时也能读到 BGE_M3_URL
load_dotenv()

# 单次 HTTP 请求的最大文本条数：与 deploy/bgem3_server.py 的 BGE_MAX_ITEMS 对应，
# 超出时在客户端自动分批，避免服务端返回 422
HTTP_BATCH_SIZE = int(os.getenv("BGE_M3_HTTP_BATCH_SIZE", "64"))
# HTTP 请求超时时间（秒），可通过环境变量调整
HTTP_TIMEOUT = float(os.getenv("BGE_M3_HTTP_TIMEOUT", "60"))

# 复用连接池的 requests 会话，避免每次请求都重新建立 TCP/TLS 连接
_session = None


def get_model_mode():
    """
    读取模型调用方式开关（环境变量 BGE_MODEL_MODE）

    :return: "http"=远端HTTP服务（默认），"local"=本地模型
    """
    mode = (os.getenv("BGE_MODEL_MODE") or "http").strip().lower()
    if mode not in ("http", "local"):
        # 配置值非法时按默认的 HTTP 模式处理，避免业务启动失败
        logger.warning(f"BGE_MODEL_MODE配置值非法：{mode}，仅支持http/local，已按http模式处理")
        return "http"
    return mode


def use_local_model():
    """
    当前是否使用本地模型推理

    :return: True=本地模型，False=远端HTTP服务
    """
    return get_model_mode() == "local"


def _get_session():
    """
    获取全局 requests 会话（懒加载单例）

    :return: 配置了自动重试策略的 requests.Session 实例
    """
    global _session
    if _session is None:
        _session = requests.Session()
        # 仅对临时性故障自动重试：连接重置、429 限流、5xx 服务端错误
        retry = Retry(
            total=2,                                   # 最多重试 2 次
            backoff_factor=0.5,                        # 重试间隔指数退避
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),       # POST 请求同样允许重试
            raise_on_status=False,                     # 重试耗尽后交回调用方统一处理
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def get_bge_m3_url():
    """
    读取远端 BGE-M3 服务地址

    :return: 完整的向量化接口 URL
    :raise RuntimeError: 未配置 BGE_M3_URL 环境变量时抛出
    """
    url = (os.getenv("BGE_M3_URL") or "").strip()
    if not url:
        logger.error("BGE-M3 服务调用失败：缺少 BGE_M3_URL 环境变量配置")
        raise RuntimeError("缺少 BGE_M3_URL 环境变量配置，无法调用远端 BGE-M3 向量服务")
    return url


def _request_embeddings(texts, mode):
    """
    向远端 BGE-M3 服务发起一次向量化请求（单批）

    :param texts: 本批待向量化的文本列表
    :param mode: 编码模式，document=文档（默认），query=查询
    :return: 已转换为 Python 原生类型、且 sparse 键为 int 的向量结果
    :raise RuntimeError: 网络异常、服务返回异常或数据结构不合法时抛出
    """
    url = get_bge_m3_url()
    start = time.perf_counter()

    try:
        # 请求体字段与服务端 EmbeddingRequest 模型保持一致
        response = _get_session().post(
            url,
            json={"embedding_documents": texts, "mode": mode},
            timeout=HTTP_TIMEOUT,
        )
        # 非 2xx 响应统一抛出 HTTPError，进入下方统一异常处理
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"BGE-M3 向量服务请求失败：{str(e)}", exc_info=True)
        raise RuntimeError(f"BGE-M3 向量服务请求失败：{str(e)}") from e

    # 解析 JSON：服务端异常时可能返回非 JSON 内容，单独捕获并给出可读报错
    try:
        data = response.json()
    except ValueError as e:
        logger.error(f"BGE-M3 向量服务返回内容不是合法 JSON：{response.text[:200]}")
        raise RuntimeError("BGE-M3 向量服务返回内容不是合法 JSON") from e

    dense = data.get("dense")
    sparse = data.get("sparse")
    # 防御性校验：dense / sparse 必须都是列表，且条数与输入文本一致
    if not isinstance(dense, list) or not isinstance(sparse, list):
        logger.error(f"BGE-M3 向量服务返回结构异常，缺少 dense/sparse 字段：{data}")
        raise RuntimeError("BGE-M3 向量服务返回结构异常：dense/sparse 字段缺失")
    if len(dense) != len(texts) or len(sparse) != len(texts):
        logger.error(
            f"BGE-M3 向量服务返回条数异常：输入 {len(texts)} 条，"
            f"dense {len(dense)} 条，sparse {len(sparse)} 条"
        )
        raise RuntimeError("BGE-M3 向量服务返回条数与输入文本条数不一致")

    try:
        # dense 统一转为 Python float 列表，保证可 JSON 序列化/可入库
        processed_dense = [[float(value) for value in vector] for vector in dense]
        # JSON 对象的 key 必须是字符串，服务端返回的稀疏向量键是 str；
        # Milvus 稀疏向量要求 token_id 为 int，这里还原为 int 键
        processed_sparse = [
            {int(token_id): float(weight) for token_id, weight in item.items()}
            for item in sparse
        ]
    except (TypeError, ValueError, AttributeError) as e:
        logger.error(f"BGE-M3 向量服务返回数据格式无法转换：{str(e)}", exc_info=True)
        raise RuntimeError(f"BGE-M3 向量服务返回数据格式无法转换：{str(e)}") from e

    cost_ms = (time.perf_counter() - start) * 1000
    logger.debug(f"BGE-M3 HTTP 向量生成完成：{len(texts)} 条，耗时 {cost_ms:.1f}ms")
    return {"dense": processed_dense, "sparse": processed_sparse}


def generate_embeddings(texts, mode="document"):
    """
    为文本列表生成稠密 + 稀疏混合向量（通过远端 HTTP 服务）

    与本地模型版 generate_embeddings 的调用方式和返回结构保持完全一致，
    现有业务代码只需替换 import 即可无缝迁移。

    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :param mode: 编码模式，document=文档（默认），query=查询
    :return: {"dense": [[float, ...], ...], "sparse": [{int: float, ...}, ...]}
    :raise ValueError: 入参不合法时抛出
    :raise RuntimeError: 远端服务调用失败时抛出
    """
    # 入参合法性校验，与原本地工具保持一致
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if mode not in ("document", "query"):
        logger.warning(f"生成向量入参不合法，mode仅支持document/query，当前为：{mode}")
        raise ValueError("参数mode仅支持document或query")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"生成向量入参不合法，第{index}条文本为空或不是字符串")
            raise ValueError(f"参数texts中第{index}条文本必须是非空字符串")

    # 根据环境变量决定调用方式：http=远端HTTP服务（默认），local=本地模型
    if use_local_model():
        # 延迟导入：仅本地模式才导入本地模型依赖（torch/FlagEmbedding），HTTP模式无需加载
        from utils.lm.embedding_utils import generate_embeddings as _local_generate_embeddings

        if mode == "query":
            # 原本地工具只提供文档编码（encode_documents），保持兼容时提示并降级
            logger.warning("本地模型工具不支持query编码模式，已按document模式编码")
        logger.info(f"BGE_MODEL_MODE=local，使用本地模型为{len(texts)}条文本生成向量")
        return _local_generate_embeddings(texts)

    logger.info(f"BGE_MODEL_MODE=http，通过HTTP服务为{len(texts)}条文本生成混合向量嵌入，mode={mode}")

    all_dense = []
    all_sparse = []
    # 按服务端单次请求上限分批，保证返回结果顺序与输入文本顺序严格一致
    for start in range(0, len(texts), HTTP_BATCH_SIZE):
        batch = texts[start:start + HTTP_BATCH_SIZE]
        result = _request_embeddings(batch, mode)
        all_dense.extend(result["dense"])
        all_sparse.extend(result["sparse"])

    logger.success(f"{len(texts)}条文本向量生成完成（HTTP），dense/sparse 条数均为 {len(all_dense)}")
    return {"dense": all_dense, "sparse": all_sparse}
