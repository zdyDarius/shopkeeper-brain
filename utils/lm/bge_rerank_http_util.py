"""
BGE Reranker HTTP 客户端工具

用于替换本地模型版工具（utils/lm/reranker_utils.py），
把「加载本地 FlagReranker」改为「请求远端 BGE Reranker HTTP 服务」。

服务端实现见 deploy/bge_reranker_server.py，接口约定：
    POST {BGE_RERANKER_URL}
    请求体：{"query": "查询文本", "documents": ["文档1", "文档2", ...]}
    响应体：{"scores": [0.87, 0.12]}   # 已做 Sigmoid 归一化，与 documents 一一对应

调用方式由环境变量 BGE_MODEL_MODE 控制：
    http  = 走远端 HTTP 服务（默认，推荐）
    local = 走本地模型（复用 utils/lm/reranker_utils.py）

对外接口保持与原工具兼容：
    get_reranker_model()                   -> 可直接调用 compute_score(sentence_pairs, normalize=True)
    compute_token_number(content)          -> 计算文本对应的 token 数量
"""

import math
import os
import threading
import time
from collections import OrderedDict

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common.config.reranker_config import reranker_config
from common.logging.logger import logger

# 加载 .env 配置：保证单独导入本模块时也能读到 BGE_RERANKER_URL
load_dotenv()

# 单次 HTTP 请求的最大文档条数：与 deploy/bge_reranker_server.py 的 BGE_RERANKER_MAX_ITEMS 对应，
# 超出时在客户端自动分批
HTTP_BATCH_SIZE = int(os.getenv("BGE_RERANKER_HTTP_BATCH_SIZE", "64"))
# HTTP 请求超时时间（秒），可通过环境变量调整
HTTP_TIMEOUT = float(os.getenv("BGE_RERANKER_HTTP_TIMEOUT", "60"))
# 无本地分词器时的兜底换算比例：BGE 系列分词器经验值约 1 个字符 ≈ 1.3 个 token
CHARS_TO_TOKEN_RATIO = 1.3
# 反 Sigmoid 时的数值裁剪边界，避免 log(0) 导致无穷大
_SCORE_EPS = 1e-7

# 复用连接池的 requests 会话，避免每次请求都重新建立 TCP/TLS 连接
_session = None
# HTTP 客户端单例，保证业务侧重复调用时复用同一实例
_reranker_client = None
_client_lock = threading.Lock()

# 分词器单例：仅用于本地精确计算 token 数，不加载重排序模型本身
_tokenizer = None
_tokenizer_lock = threading.Lock()
# 标记是否已经提示过「分词器不可用，改用估算」的告警，避免日志刷屏
_tokenizer_fallback_warned = False


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


def get_bge_reranker_url():
    """
    读取远端 BGE Reranker 服务地址

    :return: 完整的重排序接口 URL
    :raise RuntimeError: 未配置 BGE_RERANKER_URL 环境变量时抛出
    """
    url = (os.getenv("BGE_RERANKER_URL") or "").strip()
    if not url:
        logger.error("BGE Reranker 服务调用失败：缺少 BGE_RERANKER_URL 环境变量配置")
        raise RuntimeError("缺少 BGE_RERANKER_URL 环境变量配置，无法调用远端 BGE Reranker 服务")
    return url


def _inverse_sigmoid(score):
    """
    对服务端返回的 Sigmoid 归一化分数做反变换，还原为原始 logit

    服务端固定使用 normalize=True 输出 (0, 1) 分数；
    当调用方传入 normalize=False 时，通过该函数还原为原始 logit，
    保持与原本地 FlagReranker.compute_score(normalize=False) 的数值语义一致。

    :param score: (0, 1) 区间的归一化分数
    :return: 原始 logit
    """
    # 裁剪到 (eps, 1-eps)，避免 0/1 边界导致 log 运算溢出
    clipped = min(max(float(score), _SCORE_EPS), 1.0 - _SCORE_EPS)
    return math.log(clipped / (1.0 - clipped))


class BgeRerankHttpClient:
    """
    BGE Reranker HTTP 客户端

    兼容本地 FlagReranker 最常用的 compute_score(sentence_pairs, normalize=True) 调用方式，
    业务代码无需感知底层已从本地推理切换为远端 HTTP 服务。
    """

    def __init__(self, url=None):
        """
        :param url: 重排序服务地址；不传时每次请求动态读取 BGE_RERANKER_URL
        """
        self._url = url

    def _resolve_url(self):
        """解析本次请求使用的服务地址（优先使用构造时显式传入的 URL）"""
        return self._url or get_bge_reranker_url()

    def _request_scores(self, query, documents):
        """
        向远端服务发起一次打分请求（单个 query + 一批 documents）

        :param query: 查询文本
        :param documents: 本批候选文档列表
        :return: 与 documents 顺序一一对应的分数列表
        :raise RuntimeError: 网络异常、服务返回异常或数据结构不合法时抛出
        """
        url = self._resolve_url()
        start = time.perf_counter()

        try:
            # 请求体字段与服务端 RerankRequest 模型保持一致
            response = _get_session().post(
                url,
                json={"query": query, "documents": documents},
                timeout=HTTP_TIMEOUT,
            )
            # 非 2xx 响应统一抛出 HTTPError，进入下方统一异常处理
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"BGE Reranker 服务请求失败：{str(e)}", exc_info=True)
            raise RuntimeError(f"BGE Reranker 服务请求失败：{str(e)}") from e

        # 解析 JSON：服务端异常时可能返回非 JSON 内容，单独捕获并给出可读报错
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"BGE Reranker 服务返回内容不是合法 JSON：{response.text[:200]}")
            raise RuntimeError("BGE Reranker 服务返回内容不是合法 JSON") from e

        scores = data.get("scores")
        # 防御性校验：scores 必须为列表，且条数与输入文档一致，避免分数与文档错位
        if not isinstance(scores, list):
            logger.error(f"BGE Reranker 服务返回结构异常，缺少 scores 字段：{data}")
            raise RuntimeError("BGE Reranker 服务返回结构异常：scores 字段缺失")
        if len(scores) != len(documents):
            logger.error(
                f"BGE Reranker 服务返回条数异常：输入 {len(documents)} 篇文档，"
                f"返回 {len(scores)} 个分数"
            )
            raise RuntimeError("BGE Reranker 服务返回分数条数与输入文档条数不一致")

        cost_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"BGE Reranker HTTP 打分完成：{len(documents)} 篇文档，耗时 {cost_ms:.1f}ms")
        return [float(score) for score in scores]

    def compute_score(self, sentence_pairs, normalize=True, batch_size=None):
        """
        对「查询 + 候选文档」句子对逐对打分，接口与本地 FlagReranker.compute_score 兼容

        服务端接口是「单个 query + 一批 documents」，而本地接口允许句子对的 query 各不相同，
        因此这里按 query 分组后分别请求，最后按原始顺序回填分数；同组内再按批大小切分。

        :param sentence_pairs: 句子对列表，形如 [[query, document], ...]
        :param normalize: 是否返回 Sigmoid 归一化分数；服务端固定归一化，
                          传 False 时会在客户端反 Sigmoid 还原为原始 logit
        :param batch_size: 单次请求的文档条数，默认使用环境变量/常量配置
        :return: 分数列表，与 sentence_pairs 顺序严格一一对应
        :raise ValueError: 入参不合法时抛出
        :raise RuntimeError: 远端服务调用失败时抛出
        """
        # 入参校验：必须是非空句子对列表，且每项为 [query, document]
        if not isinstance(sentence_pairs, list) or len(sentence_pairs) == 0:
            logger.warning("重排序入参不合法，sentence_pairs必须为非空列表")
            raise ValueError("参数sentence_pairs必须是包含句子对的非空列表")

        pairs = []
        for index, pair in enumerate(sentence_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                logger.warning(f"重排序入参不合法，第{index}项不是[query, document]结构")
                raise ValueError(f"参数sentence_pairs中第{index}项必须是[query, document]结构")
            query, document = pair
            # 与原本地实现一致：将 None 兜底为空串，其余统一转字符串
            pairs.append((str(query or ""), str(document or "")))

        # 按 query 分组并保留组内原始下标，便于请求完成后按原顺序回填分数
        groups = OrderedDict()
        for index, (query, document) in enumerate(pairs):
            groups.setdefault(query, []).append((index, document))

        batch_size = batch_size or HTTP_BATCH_SIZE
        scores = [0.0] * len(pairs)

        for query, items in groups.items():
            # 单次请求文档数不能超过服务端上限，超出时分批请求
            for start in range(0, len(items), batch_size):
                batch = items[start:start + batch_size]
                documents = [document for _, document in batch]
                batch_scores = self._request_scores(query, documents)
                for (index, _), score in zip(batch, batch_scores):
                    scores[index] = score

        # 服务端固定返回归一化分数；调用方要求原始 logit 时，客户端做反 Sigmoid 还原
        if not normalize:
            scores = [_inverse_sigmoid(score) for score in scores]

        return scores


def get_reranker_model():
    """
    获取重排序 HTTP 客户端单例

    保持与原本地工具相同的函数名，业务侧 get_reranker_model().compute_score(...) 无需改动。
    当 BGE_MODEL_MODE=local 时，直接返回本地 FlagReranker 模型。

    :return: BgeRerankHttpClient 实例（HTTP模式）或 FlagReranker 实例（本地模式）
    """
    # 本地模式：复用原有本地模型工具，保持调用方代码完全不变
    if use_local_model():
        from utils.lm.reranker_utils import get_reranker_model as _local_get_reranker_model

        logger.info("BGE_MODEL_MODE=local，使用本地重排序模型")
        return _local_get_reranker_model()

    global _reranker_client
    if _reranker_client is None:
        with _client_lock:
            # 双重检查：等待锁期间可能已被其他线程初始化完成
            if _reranker_client is None:
                _reranker_client = BgeRerankHttpClient()
                logger.info("BGE Reranker HTTP 客户端单例初始化完成")
    return _reranker_client


def _get_tokenizer():
    """
    获取本地分词器（懒加载单例），仅用于精确计算 token 数

    说明：远端服务未提供分词接口，这里优先复用 BGE_RERANKER_LARGE 指向的本地模型目录中的
    tokenizer（体积极小，不会加载模型权重）；本地目录不存在或加载失败时返回 None，
    由 compute_token_number 走字符数估算兜底。

    :return: transformers 分词器实例，不可用时返回 None
    """
    global _tokenizer, _tokenizer_fallback_warned

    if _tokenizer is not None:
        return _tokenizer

    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer

        model_path = reranker_config.bge_reranker_large
        # 仅当配置的是真实存在的本地目录时才加载，避免触发联网下载
        if model_path and os.path.isdir(model_path):
            try:
                from transformers import AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                logger.debug(f"本地分词器加载成功，用于token数精确计算：{model_path}")
                return _tokenizer
            except Exception as e:
                logger.warning(f"本地分词器加载失败，将使用字符数估算token：{str(e)}")

        # 仅提示一次，避免批量调用时日志刷屏
        if not _tokenizer_fallback_warned:
            _tokenizer_fallback_warned = True
            logger.warning(
                "未找到可用的本地分词器（BGE_RERANKER_LARGE），"
                f"compute_token_number 将按 {CHARS_TO_TOKEN_RATIO} 倍字符数估算token"
            )
        return None


def compute_token_number(content: str):
    """
    计算字符串对应的 token 数量

    优先使用本地 BGE 分词器精确计算；本地分词器不可用时，按
    1 个字符 ≈ 1.3 个 token 的经验比例向上取整估算，保证不会低估导致超限。

    :param content: 待计算文本
    :return: token 数量
    """
    # 与原本地实现一致：空内容直接返回 0
    if not content:
        return 0

    # 本地模式：直接复用本地模型工具的分词器计算，保证结果与旧逻辑完全一致
    if use_local_model():
        from utils.lm.reranker_utils import compute_token_number as _local_compute_token_number

        return _local_compute_token_number(content)

    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        # add_special_tokens=False 与原本地实现保持一致
        token_list = tokenizer.encode(content, add_special_tokens=False)
        return len(token_list)

    # 兜底估算：向上取整，避免低估 token 数导致超过 reranker 的 512 窗口
    return math.ceil(len(content) * CHARS_TO_TOKEN_RATIO)
