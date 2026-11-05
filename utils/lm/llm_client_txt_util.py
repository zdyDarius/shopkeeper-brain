"""
文本大语言模型客户端工具

用于替换 utils/lm/lm_utils.py 中「文本大模型」的调用场景，
把文本模型从阿里云 DashScope 切换到自建 vLLM OpenAI 兼容服务。

注意：utils/lm/lm_utils.py 保持不变，VL 多模态模型（图片理解）仍继续使用它，
本工具只负责纯文本大模型场景，二者职责分离。

调用来源由环境变量 TEXT_LLM_PROVIDER 控制：
    vllm   = 走自建 vLLM OpenAI 兼容服务（默认）
    aliyun = 走原有阿里云 DashScope（内部直接复用 utils/lm/lm_utils.py，行为完全不变）

对外接口与原工具保持一致（get_llm_client(model=None, json_mode=False)），
返回 LangChain ChatOpenAI 实例，业务侧 `llm | StrOutputParser()`、invoke/stream 无需改动。
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

from common.logging.logger import logger

# 加载 .env 配置：保证单独导入本模块时也能读到 TEXT_LLM_* 环境变量
load_dotenv()

# 全局缓存：键为(来源, 模型名, JSON输出模式)元组，值为ChatOpenAI实例
# 作用：避免重复初始化客户端，同时区分 vllm / aliyun 两种来源
_llm_client_cache = {}


def get_text_llm_provider():
    """
    读取文本大模型调用来源开关（环境变量 TEXT_LLM_PROVIDER）

    :return: "vllm"=自建vLLM服务（默认），"aliyun"=阿里云DashScope
    """
    provider = (os.getenv("TEXT_LLM_PROVIDER") or "vllm").strip().lower()
    if provider not in ("vllm", "aliyun"):
        # 配置值非法时按默认的 vllm 模式处理，避免业务启动失败
        logger.warning(
            f"TEXT_LLM_PROVIDER配置值非法：{provider}，仅支持vllm/aliyun，已按vllm模式处理"
        )
        return "vllm"
    return provider


def _get_vllm_config():
    """
    读取自建 vLLM 服务配置

    :return: 元组(api_key, base_url, default_model, temperature)
    :raise ValueError: 缺失 API 密钥 / 服务地址等核心配置时抛出
    """
    api_key = (os.getenv("TEXT_LLM_OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("TEXT_LLM_OPENAI_API_BASE") or "").strip()
    default_model = (os.getenv("TEXT_LLM_DEFAULT_MODEL") or "").strip()

    # 温度：优先 TEXT_LLM_TEMPERATURE，其次沿用项目原有的 LLM_DEFAULT_TEMPERATURE，最后兜底 0.1
    raw_temperature = (
        os.getenv("TEXT_LLM_TEMPERATURE")
        or os.getenv("LLM_DEFAULT_TEMPERATURE")
        or "0.1"
    )
    try:
        temperature = float(raw_temperature)
    except (TypeError, ValueError):
        logger.warning(f"文本模型温度配置非法：{raw_temperature}，已按0.1处理")
        temperature = 0.1

    # 核心配置校验：提前抛出明确异常，避免调用时才报鉴权/连接错误
    if not api_key:
        raise ValueError("[文本LLM客户端] 配置缺失：请在.env中配置TEXT_LLM_OPENAI_API_KEY")
    if not base_url:
        raise ValueError("[文本LLM客户端] 配置缺失：请在.env中配置TEXT_LLM_OPENAI_API_BASE")

    return api_key, base_url, default_model, temperature


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取文本大模型客户端（带全局缓存）

    调用函数名与原有工具完全一致，业务侧只需更换 import 即可完成切换。

    :param model: 模型名称；不传时使用 TEXT_LLM_DEFAULT_MODEL（vllm模式）
                  或 lm_config.llm_model（aliyun模式）
    :param json_mode: 是否开启JSON输出模式，开启后强制模型返回标准json_object
    :return: ChatOpenAI 实例（优先从全局缓存获取）
    :raise ValueError: 缺失核心配置时抛出
    :raise Exception: 模型初始化失败（LangChain封装层异常）
    """
    provider = get_text_llm_provider()

    # 阿里云模式：完全复用原工具，保证原有配置、模型与调用行为不变
    if provider == "aliyun":
        # 延迟导入：vllm 模式无需加载阿里云相关配置与客户端
        from utils.lm.lm_utils import get_llm_client as _aliyun_get_llm_client

        logger.debug("[文本LLM客户端] TEXT_LLM_PROVIDER=aliyun，复用阿里云文本模型客户端")
        return _aliyun_get_llm_client(model=model, json_mode=json_mode)

    # vllm 模式：使用自建服务的独立配置
    api_key, base_url, default_model, temperature = _get_vllm_config()
    # 模型名优先级：传入参数 > TEXT_LLM_DEFAULT_MODEL；均未配置时给出明确报错
    target_model = (model or "").strip() or default_model
    if not target_model:
        raise ValueError(
            "[文本LLM客户端] 配置缺失：请传入model或在.env中配置TEXT_LLM_DEFAULT_MODEL"
        )

    # 缓存键加入来源，避免运行期切换来源时误用另一种来源的客户端
    cache_key = (provider, target_model, json_mode)
    if cache_key in _llm_client_cache:
        logger.debug(
            f"[文本LLM客户端] 缓存命中，直接返回实例：来源={provider}，模型={target_model}，JSON模式={json_mode}"
        )
        return _llm_client_cache[cache_key]

    logger.info(
        f"[文本LLM客户端] 开始初始化新实例：来源={provider}，模型={target_model}，JSON模式={json_mode}"
    )

    # model_kwargs：OpenAI通用参数，vLLM 的 OpenAI 兼容接口同样支持
    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug("[文本LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # extra_body：通过 chat_template_kwargs 关闭 Qwen3 的思考链输出，
    # 与原有阿里云客户端的 enable_thinking=False 行为保持一致，减少冗余推理内容
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        llm_client = ChatOpenAI(
            model=target_model,          # 自建服务上的模型名，如 qwen3-8b
            temperature=temperature,      # 低温度保证输出确定性
            api_key=api_key,              # 自建服务的 API Key
            base_url=base_url,            # 自建 vLLM 服务的 /v1 地址
            extra_body=extra_body,        # 关闭思考链（Qwen3 + vLLM）
            model_kwargs=model_kwargs,    # JSON输出等通用参数
        )
    except LangChainException as e:
        raise Exception(
            f"[文本LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}"
        ) from e

    # 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = llm_client
    logger.info(
        f"[文本LLM客户端] 实例初始化成功并缓存：来源={provider}，模型={target_model}，JSON模式={json_mode}"
    )
    return llm_client
