# -*- coding: utf-8 -*-
"""
基于 ragas 的 RAG 流程评估测试程序
==================================================
【程序功能】
1. 读取评估测试集 eval/qa.csv（question、ground_truth 两列）
2. 逐条调用被评估的 RAG 主流程（processor/query_processor/main_graph.py 中的 query_app）
3. 从流程结束后的 state 中提取 answer（最终答案）与 reranked_docs（重排后的检索上下文）
4. 使用项目自带的大模型客户端（utils/lm/lm_utils.py）与 BGE-M3 嵌入客户端（utils/lm/embedding_utils.py）
   作为 ragas 的评估器（裁判模型 + 嵌入模型）
5. 计算 ragas 五大指标：
   Faithfulness（答案忠实度）、Answer Relevancy（答案相关性）、
   Context Precision（上下文精确率）、Context Recall（上下文召回率）、
   Answer Correctness（答案正确性）
6. 评估结果保存到 eval/result.csv，固定 9 列：
   question、contexts、answer、ground_truth、
   Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness

【运行方式】
    cd 项目根目录
    python eval/eval.py

【前置条件】
    Milvus / MongoDB / 大模型 API / 百炼 MCP 等外部服务可用（与正常启动 RAG 服务要求一致）
    依赖已安装 ragas>=0.4（uv pip install ragas）

【版本说明】
    ragas 0.4.x 起，旧版 evaluate() 入口与旧版指标已全部标记弃用，
    新版指标从 ragas.metrics.collections 导入，构造时注入裁判模型（InstructorLLM）与嵌入模型，
    逐样本调用指标的 score() 方法即可计算得分，本程序即按新版 API 实现。
"""

import asyncio
import sys
import time
import types
import uuid
from pathlib import Path

# ======================== 项目根目录引导 ========================
# 说明：直接运行 python eval/eval.py 时，Python 只会把 eval/ 目录加入 sys.path，
# 不会包含项目根目录，因此这里手动把项目根目录插入 sys.path，保证可以导入 processor/utils 等项目包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ======================== ragas 兼容性补丁 ========================
def _patch_langchain_community_vertexai():
    """
    ragas 0.4.x 与新版 langchain-community 的兼容性补丁
    问题背景：ragas 的 llms/base.py 在模块导入时无条件执行
        from langchain_community.chat_models.vertexai import ChatVertexAI
    而项目安装的 langchain-community 0.4+ 已删除 vertexai 相关模块，
    导致 ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
    处理方式：导入失败时向 sys.modules 注入一个占位模块与占位类。
    安全性说明：ChatVertexAI 在 ragas 内部仅用于 isinstance 类型判断
    （is_multiple_completion_supported 函数），不会被实例化，占位类完全够用；
    我们传入的裁判模型是项目自己的大模型客户端，不受该占位类影响。
    """
    try:
        # 先尝试正常导入，模块存在则无需任何补丁
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
        return
    except ImportError:
        pass

    # 构造占位模块 langchain_community.chat_models.vertexai，并提供占位类 ChatVertexAI
    stub_module = types.ModuleType("langchain_community.chat_models.vertexai")

    class _StubChatVertexAI:
        """占位类：ragas 仅在 isinstance 判断中使用，永远不会被实例化"""

    stub_module.ChatVertexAI = _StubChatVertexAI
    # 注册到 sys.modules，让后续 ragas 的 from ... import ChatVertexAI 能找到该模块
    sys.modules["langchain_community.chat_models.vertexai"] = stub_module


# 执行兼容性补丁（必须在导入 ragas 之前执行）
_patch_langchain_community_vertexai()


# 第三方库导入
import pandas as pd  # 用于读写 csv 文件与结果整理

# ragas 评估框架相关导入（必须在执行完兼容性补丁之后）
from ragas.embeddings.base import BaseRagasEmbedding  # ragas 0.4 现代嵌入接口抽象基类
from ragas.llms import llm_factory  # ragas 0.4 裁判大模型工厂（结构化输出）
from ragas.metrics.collections import (  # ragas 0.4 五大评估指标（新版集合）
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# 项目内部模块导入
from common.logging.logger import logger  # 项目统一日志对象
from processor.query_processor.main_graph import query_app  # RAG 主流程编译后的可执行图
from processor.query_processor.state import create_query_default_state  # 查询流程默认状态工厂
from utils.lm.embedding_utils import generate_embeddings  # BGE-M3 向量生成工具
from utils.lm.lm_utils import get_llm_client  # 大模型客户端工具

# ======================== 全局常量配置 ========================
# 评估测试集路径
EVAL_QA_CSV = PROJECT_ROOT / "eval" / "qa.csv"
# 评估结果输出路径
EVAL_RESULT_CSV = PROJECT_ROOT / "eval" / "result.csv"
# 单个指标计算的最大尝试次数（含首次执行，重试可覆盖大模型接口偶发限流/超时）
EVAL_MAX_RETRIES = 3
# 指标计算失败后的重试等待秒数
EVAL_RETRY_WAIT_SECONDS = 2
# 多条上下文拼接为 csv 单元格文本时的分隔符
CONTEXT_SEPARATOR = "\n"


# ======================== 嵌入模型适配器 ========================
class BgeM3RagasEmbedding(BaseRagasEmbedding):
    """
    BGE-M3 嵌入模型的 ragas 现代嵌入接口适配器
    作用：把项目中 utils/lm/embedding_utils.py 的 generate_embeddings 函数包装成
          ragas 0.4 的 BaseRagasEmbedding 标准接口，供 Answer Relevancy（答案相关性）、
          Answer Correctness（答案正确性）指标在计算语义相似度时使用
    """

    def embed_text(self, text, **kwargs):
        """
        生成单条文本的嵌入向量（同步接口，ragas 计算语义相似度时调用）
        :param text: 单条文本
        :return: 稠密向量列表（浮点数列表）
        """
        # generate_embeddings 要求入参必须是列表，单条文本需先封装为列表；
        # 返回结果中 dense 字段为稠密向量嵌套列表，与输入文本一一对应
        result = generate_embeddings([text])
        return result["dense"][0]

    async def aembed_text(self, text, **kwargs):
        """
        生成单条文本的嵌入向量（异步接口，ragas 并发计算时调用）
        :param text: 单条文本
        :return: 稠密向量列表（浮点数列表）
        说明：BGE-M3 是本地同步推理模型，无原生异步接口，这里直接复用同步实现
        """
        # 直接调用同步实现（模型内部为单例对象，重复调用安全）
        return self.embed_text(text)


# ======================== 核心步骤函数 ========================
def step_1_load_eval_dataset(csv_path=EVAL_QA_CSV):
    """
    核心步骤1：读取评估测试集 qa.csv
    :param csv_path: 测试集文件路径，默认 eval/qa.csv
    :return: 只包含 question、ground_truth 两列且无空值的 DataFrame
    说明：使用 utf-8-sig 编码读取，自动去除 csv 文件开头的 BOM 头（\\ufeff），
          避免第一列列名被读成 '\\ufeffquestion' 导致后续按列名取值失败
    """
    # 读取 csv 文件（utf-8-sig 兼容带 BOM 的 utf-8 文件）
    qa_df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 校验必需的两列是否存在
    for column in ("question", "ground_truth"):
        if column not in qa_df.columns:
            raise ValueError(f"评估测试集缺少必需列：{column}，实际列名：{list(qa_df.columns)}")

    # 只保留必需的两列，并丢弃问题或标准答案为空的行
    qa_df = qa_df[["question", "ground_truth"]].dropna(subset=["question", "ground_truth"]).reset_index(drop=True)

    logger.info(f"核心步骤1：评估测试集加载完成，共 {len(qa_df)} 条待评估样本")
    return qa_df


def step_2_run_rag_pipeline(question):
    """
    核心步骤2：调用被评估的 RAG 主流程，获取最终答案与检索上下文
    :param question: 用户原始问题
    :return: (answer, contexts) 元组
             answer —— RAG 流程生成的最终答案（state 中的 answer）
             contexts —— 重排后的上下文文本列表（state 中 reranked_docs 每个文档的 text 字段）
    """
    # 每个问题使用独立的随机 session_id，保证多轮评估互不干扰
    # （session_id 会影响 MongoDB 历史记录的读取，独立 id 可让每次评估在“无历史”条件下进行，结果更稳定）
    session_id = f"eval_{uuid.uuid4().hex[:12]}"

    # 构造查询流程的初始状态（与 api/query_service.py 中的调用方式保持一致）
    state = create_query_default_state(
        session_id=session_id,
        is_stream=False,  # 评估场景统一使用非流式输出，直接拿到完整答案
        original_query=question,
    )

    # 调用 RAG 主流程，得到包含全部中间结果的最终状态
    final_state = query_app.invoke(state)

    # 提取最终答案
    answer = final_state.get("answer") or ""

    # 提取重排后的 Top-K 文档，只取每个文档的 text 字段作为 ragas 的上下文
    reranked_docs = final_state.get("reranked_docs") or []
    contexts = [doc.get("text") or "" for doc in reranked_docs]

    return answer, contexts


def step_3_build_ragas_samples(records):
    """
    核心步骤3：把 RAG 运行结果整理成 ragas 标准命名的评估样本列表
    :param records: 每条样本的字典列表，包含 question、contexts、answer、ground_truth 四个字段
    :return: ragas 标准命名的样本字典列表，字段含义如下：
             user_input —— 用户问题
             retrieved_contexts —— 检索上下文列表
             response —— RAG 生成的答案
             reference —— 人工标注的标准答案
    """
    # 按照各指标的必须字段做非空过滤（缺字段的样本无法参与任何指标计算）
    samples = [
        {
            "user_input": record["question"],
            "retrieved_contexts": record["contexts"],
            "response": record["answer"],
            "reference": record["ground_truth"],
        }
        for record in records
    ]

    logger.info(f"核心步骤3：ragas 评估样本整理完成，共 {len(samples)} 条样本")
    return samples


def step_4_build_evaluator():
    """
    核心步骤4：构建 ragas 评估器（裁判大模型 + 嵌入模型）
    :return: (evaluator_llm, evaluator_embeddings) 元组
             evaluator_llm —— 基于项目大模型客户端构建的 ragas 裁判模型（结构化输出）
             evaluator_embeddings —— 基于项目 BGE-M3 嵌入工具构建的 ragas 嵌入模型
    """
    # 获取项目的大模型客户端（ChatOpenAI，与被评估的 RAG 流程使用同一个模型、密钥与接口地址）
    llm_client = get_llm_client()

    # 取出 ChatOpenAI 底层封装的原生异步 OpenAI 客户端（自动携带项目配置的 api_key 与 base_url）
    # 必须使用异步客户端：ragas 0.4 的 collections 指标内部统一调用 agenerate 异步生成，
    # 同步客户端会直接报错“Cannot use agenerate() with a synchronous client”
    openai_async_client = llm_client.root_async_client

    # 构建裁判大模型：模型名称直接取自项目大模型客户端，保证“裁判”与“被评流程”使用同一个模型；
    # llm_factory 内部会用 instructor 把异步客户端包装成 AsyncInstructor（支持结构化 JSON 输出）
    # max_tokens 调大到 2048：ragas 默认 1024，长答案拆分论断+逐条裁决的 JSON 输出会被截断，
    # 导致报错“The output is incomplete due to a max_tokens length limit”（该参数会透传给底层模型）
    evaluator_llm = llm_factory(
        model=llm_client.model_name,
        provider="openai",
        client=openai_async_client,
        max_tokens=2048,
    )

    # 构建嵌入评估器：复用项目的 BGE-M3 嵌入工具（本地模型单例，只会加载一次）
    evaluator_embeddings = BgeM3RagasEmbedding()

    logger.info(f"核心步骤4：ragas 评估器构建完成（裁判大模型：{llm_client.model_name} + BGE-M3 嵌入模型）")
    return evaluator_llm, evaluator_embeddings


async def _step_5_score_metric_with_retry(metric_name, metric, sample_kwargs, sample_index):
    """
    核心步骤5的内部协程：带重试地计算单个样本的单个指标得分
    :param metric_name: 指标名（用于日志输出）
    :param metric: 已实例化的 ragas collections 指标对象
    :param sample_kwargs: 该指标计算所需的样本字段字典
    :param sample_index: 样本序号（从 0 开始，用于日志输出）
    :return: 指标得分（0~1 的浮点数），全部重试失败时返回 None
    """
    # 带重试的指标计算：失败自动重试，全部失败后返回 None（最终写入结果时表现为 NaN）
    for attempt in range(1, EVAL_MAX_RETRIES + 1):
        try:
            # 调用指标的异步计算接口 ascore，返回 MetricResult 对象
            metric_result = await metric.ascore(**sample_kwargs)
            logger.debug(f"第 {sample_index + 1} 条样本指标 {metric_name} 计算详情：{metric_result.reason}")
            # MetricResult.value 即 0~1 的指标得分
            return metric_result.value
        except Exception as e:  # noqa: BLE001 评估场景需要捕获所有异常保证流程不中断
            logger.warning(
                f"第 {sample_index + 1} 条样本指标 {metric_name} 第 {attempt}/{EVAL_MAX_RETRIES} 次计算失败：{e}"
            )
            # 还有重试机会则等待后重试（异步等待，不阻塞事件循环）
            if attempt < EVAL_MAX_RETRIES:
                await asyncio.sleep(EVAL_RETRY_WAIT_SECONDS)
    # 全部重试均失败
    return None


async def _step_5_evaluate_all(metrics, samples):
    """
    核心步骤5的内部协程：在同一个事件循环中完成全部样本 × 全部指标的评估
    :param metrics: 指标定义列表，元素为 (指标名, 指标对象, 必需样本字段元组)
    :param samples: ragas 标准命名的样本字典列表
    :return: 每条样本的得分行列表（键为指标名，值为得分或 None）
    说明：必须在同一个事件循环中完成全部计算——底层 HTTP 客户端（AsyncOpenAI/httpx）
          绑定到事件循环，跨多个事件循环复用同一客户端会导致“Event loop is closed”报错
    """
    # 逐样本计算五大指标得分（串行执行，避免触发大模型接口限流）
    score_rows = []
    for i, sample in enumerate(samples):
        # 当前样本的得分行（键为指标名，值为得分或 None）
        score_row = {}
        for metric_name, metric, required_fields in metrics:
            # 前置校验：该指标必需的字段为空（如 RAG 流程失败导致答案/上下文为空）时直接记 None，
            # 避免把必然失败的请求发给大模型
            if any(not sample.get(field) for field in required_fields):
                logger.warning(f"第 {i + 1} 条样本字段为空，指标 {metric_name} 无法计算，记为 NaN")
                score_row[metric_name] = None
                continue

            # 按指标声明的必需字段构造入参（不同指标需要的字段不同）
            sample_kwargs = {field: sample[field] for field in required_fields}

            # 带重试地计算当前指标的得分
            score_row[metric_name] = await _step_5_score_metric_with_retry(
                metric_name, metric, sample_kwargs, i
            )

        score_rows.append(score_row)
        logger.info(f"核心步骤5：第 {i + 1}/{len(samples)} 条样本五大指标计算完成：{score_row}")

    return score_rows


def step_5_run_evaluate(samples, evaluator_llm, evaluator_embeddings):
    """
    核心步骤5：逐样本执行 ragas 五大指标评估
    :param samples: 核心步骤3 整理好的 ragas 标准命名样本列表
    :param evaluator_llm: 核心步骤4 构建的裁判大模型
    :param evaluator_embeddings: 核心步骤4 构建的嵌入模型
    :return: 评估结果 DataFrame，行顺序与 samples 完全一致，
             列为五大指标的 ragas 指标名（faithfulness、answer_relevancy、
             context_precision、context_recall、answer_correctness），
             单条指标计算失败时对应位置记为 NaN
    说明：为避免触发大模型接口限流，采用“逐样本 × 逐指标”串行计算，
          每个指标失败后自动重试（最多 EVAL_MAX_RETRIES 次）。
    """
    # 实例化 ragas 0.4 五大评估指标（构造时注入裁判模型与嵌入模型），并声明各指标计算所需的样本字段：
    #   Faithfulness —— 需要：问题 + 答案 + 检索上下文
    #   AnswerRelevancy —— 需要：问题 + 答案（借助嵌入计算答案与问题的语义相关性）
    #   ContextPrecision —— 需要：问题 + 标准答案 + 检索上下文（评估上下文排序质量）
    #   ContextRecall —— 需要：问题 + 标准答案 + 检索上下文（评估上下文对标准答案的覆盖度）
    #   AnswerCorrectness —— 需要：问题 + 答案 + 标准答案（事实一致性 + 嵌入语义相似度加权）
    metrics = [
        ("faithfulness", Faithfulness(evaluator_llm), ("user_input", "response", "retrieved_contexts")),
        ("answer_relevancy", AnswerRelevancy(evaluator_llm, evaluator_embeddings), ("user_input", "response")),
        ("context_precision", ContextPrecision(evaluator_llm), ("user_input", "reference", "retrieved_contexts")),
        ("context_recall", ContextRecall(evaluator_llm), ("user_input", "reference", "retrieved_contexts")),
        ("answer_correctness", AnswerCorrectness(evaluator_llm, evaluator_embeddings), ("user_input", "response", "reference")),
    ]

    # 在单个事件循环中执行全部评估计算（保证底层 HTTP 客户端只在同一个事件循环中使用）
    score_rows = asyncio.run(_step_5_evaluate_all(metrics, samples))

    # 组装为 DataFrame（行顺序与 samples 一致，便于后续与原始记录对齐）
    result_df = pd.DataFrame(score_rows, columns=[m[0] for m in metrics])
    logger.info("核心步骤5：ragas 五大指标评估执行完成")
    return result_df


def step_6_save_result(records, result_df, out_path=EVAL_RESULT_CSV):
    """
    核心步骤6：整理评估结果并保存到 eval/result.csv（固定 9 列）
    :param records: 核心步骤2 收集的原始运行记录（保证 question/contexts/answer/ground_truth 列输出）
    :param result_df: 核心步骤5 得到的指标得分 DataFrame（行顺序与 records 一致）
    :param out_path: 结果输出路径，默认 eval/result.csv
    :return: 整理好的 9 列结果 DataFrame
    """
    # 目标列名 -> ragas 结果中可能出现的列名列表（兼容不同版本 ragas 的命名差异）
    metric_columns = {
        "Faithfulness": ["faithfulness", "Faithfulness"],
        "Answer Relevancy": ["answer_relevancy", "Answer Relevancy"],
        "Context Precision": ["context_precision", "Context Precision"],
        "Context Recall": ["context_recall", "Context Recall"],
        "Answer Correctness": ["answer_correctness", "Answer Correctness"],
    }

    # 结果行数与运行记录数必须一致，避免错位写入
    if len(result_df) != len(records):
        raise ValueError(f"评估结果行数（{len(result_df)}）与样本记录数（{len(records)}）不一致，禁止写入结果文件")

    # 逐行组装最终输出内容
    out_rows = []
    for i, record in enumerate(records):
        # 前 4 列：直接取自 RAG 运行记录，contexts 多条文本拼接为一个单元格
        row = {
            "question": record["question"],
            "contexts": CONTEXT_SEPARATOR.join(record["contexts"]),
            "answer": record["answer"],
            "ground_truth": record["ground_truth"],
        }
        # 后 5 列：按列名映射从 ragas 结果中取指标得分
        for target_name, candidates in metric_columns.items():
            value = None
            for column in candidates:
                if column in result_df.columns:
                    value = result_df.iloc[i].get(column)
                    break
            row[target_name] = value
        out_rows.append(row)

    # 固定 9 列列名，保证输出结构稳定
    out_df = pd.DataFrame(out_rows, columns=[
        "question",
        "contexts",
        "answer",
        "ground_truth",
        "Faithfulness",
        "Answer Relevancy",
        "Context Precision",
        "Context Recall",
        "Answer Correctness",
    ])

    # 写入 csv（utf-8-sig 带 BOM，保证 Excel 直接打开不乱码）
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"核心步骤6：评估结果已保存到 {out_path}")

    return out_df


# ======================== 程序主入口 ========================
def main():
    """
    程序主入口：按顺序串联六大核心步骤，完成整个评估流程
    """
    # 记录流程开始时间，用于统计总耗时
    start_time = time.time()

    # 核心步骤1：读取评估测试集
    qa_df = step_1_load_eval_dataset()

    # 核心步骤2：逐条运行 RAG 流程，收集答案与上下文
    records = []
    for i, row in qa_df.iterrows():
        question = str(row["question"])
        ground_truth = str(row["ground_truth"])
        logger.info(f"核心步骤2：开始评估第 {i + 1}/{len(qa_df)} 条样本，问题：{question}")

        # 单条样本执行失败不中断整体评估，记录空答案与空上下文后继续
        try:
            answer, contexts = step_2_run_rag_pipeline(question)
        except Exception as e:  # noqa: BLE001 评估场景需要捕获所有异常保证流程不中断
            logger.error(f"第 {i + 1} 条样本 RAG 流程执行失败：{e}")
            answer, contexts = "", []

        records.append(
            {
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": ground_truth,
            }
        )

    # 核心步骤3：整理成 ragas 标准命名的评估样本
    samples = step_3_build_ragas_samples(records)

    # 核心步骤4：构建评估器（裁判大模型 + 嵌入模型）
    evaluator_llm, evaluator_embeddings = step_4_build_evaluator()

    # 核心步骤5：执行 ragas 五大指标评估
    result_df = step_5_run_evaluate(samples, evaluator_llm, evaluator_embeddings)

    # 核心步骤6：整理并保存评估结果
    out_df = step_6_save_result(records, result_df)

    # 打印五大指标的平均得分，快速了解 RAG 流程整体质量
    for column in ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall", "Answer Correctness"]:
        logger.info(f"指标平均得分 {column}: {out_df[column].mean():.4f}")

    # 统计并输出总耗时
    elapsed = time.time() - start_time
    logger.info(f"评估流程全部完成，总耗时 {elapsed:.1f} 秒，结果文件：{EVAL_RESULT_CSV}")


# 脚本直接运行时执行主流程
if __name__ == "__main__":
    main()
