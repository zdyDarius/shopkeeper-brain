import json
import os
import sys
from email import message
from lib2to3.fixes.fix_input import context
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from multipart import file_path
from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.task_utils import add_running_task, add_done_task

# 主体识别上下文切片数：取前 K 个切片用于 LLM 识别
ITEM_NAME_CONTEXT_CHUNK_K = 5
# 主体识别上下文总字符数上限：防止上下文过长导致大模型输入超限
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS = 2000

@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """

    # 1: 添加节点到运行时列表 add_running_task(state.get("task_id"), "node_item_name_recognition")
    add_running_task(state.get("task_id"), "node_item_name_recognition")
    # 2: 获取并校验参数 chunks file_title
    chunks, file_title = step_1_validate_and_get_data(state)
    # 3: 使用模型提取item_name(chunks file_title) -> item_name
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)
    # 4: 向chunk中回填item_name属性
    step_3_padding_item_name_to_chunks(chunks, item_name)

    # 5. 在milvus中创建存储item_name的collection
    step_4_prepared_item_name_collection()
    # 6. 将数据写到milvus的collection中
    step_5_insert_item_name_data(item_name, file_title)
    # 7. 更新状态
    state["item_name"] = item_name
    state["chunks"] = chunks
    #  添加节点到已完成列表
    add_done_task(state.get("task_id"), "node_item_name_recognition")
    return state


    return state
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState):
    chunks = state.get("chunks")
    file_title = state.get("file_title")
    md_path = state.get("md_path")
    md_path_obj: Path = Path(md_path)
    if not chunks :
        if md_path_obj.is_file() :
            json_path_obj:Path = md_path_obj.parent / f'{md_path_obj.stem}.json'
            if json_path_obj.is_file() :
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
            else:
                logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
                raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
    if not file_title:
        file_title = md_path_obj.stem or "default_title"
        state['file_title'] = file_title
        logger.warning(f"file_title不存在,给与默认值:{file_title}")

    return chunks, file_title

@step_log("step_2_call_llm_return_item_name")
def step_2_call_llm_return_item_name(chunks, file_title):
    llm_model = get_llm_client()
    context = ''
    for chunk in chunks [:ITEM_NAME_CONTEXT_CHUNK_K]:
        context +=f"标题: {chunk.get('parent_title')} , 内容: {chunk.get('content')} \n"

    context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    prompt_text = load_prompt(name='item_name_recognition',file_title=file_title,context=context)

    message = HumanMessage(content=prompt_text)

    chains = llm_model | StrOutputParser()
    item_name = chains.invoke([message])
    if not item_name:
        item_name = file_title
        logger.warning(f"没有识别出item_name,使用file_title赋值:{item_name}")
    logger.debug(item_name)

    return item_name

@step_log("step_3_padding_item_name_to_chunks")
def step_3_padding_item_name_to_chunks(chunks, item_name):
    for chunk in chunks:
        chunk["item_name"] = item_name
        logger.debug(chunk)

@step_log("step_4_prepared_item_name_collection")
def step_4_prepared_item_name_collection():
    # 获取操作milvus数据库的客户端
    milvus_client = get_milvus_client()
    has_collection = milvus_client.has_collection(collection_name=milvus_config.item_name_collection)
    if has_collection:
        logger.info(f"{milvus_config.item_name_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.item_name_collection}不存在,进行集合的创建!")
    # 创建schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    # 通过schema维护表中的字段(field)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    # 创建索引参数   索引作用：提高查询性能
    index_params = milvus_client.prepare_index_params()
    # 添加索引---稠密字段
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 64,
            "efConstruction": 100
        }
    )
    # 添加索引---稀疏字段
    index_params.add_index(
        field_name="sparse_vector",  # 字段名称
        index_name="sparse_vector_index",  # 索引名称
        index_type="SPARSE_INVERTED_INDEX",  # 索引类型
        metric_type="IP",  # 算分方式
        params={"inverted_index_algo": "DAAT_MAXSCORE"}  # 跳过低分，只算高分
    )

    # 创建collection
    milvus_client.create_collection(
        collection_name=milvus_config.item_name_collection,
        schema=schema,
        index_params=index_params
    )
    logger.success(f"集合{milvus_config.item_name_collection}创建成功")


@step_log("step_5_insert_item_name_data")
def step_5_insert_item_name_data(item_name, file_title):
    """将模型识别出来主题名写到milvus向量数据库中"""
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 删除数据
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        # 注意：  1.  等值比较用==     2.  file_tile 用引号括起来
        filter=f"file_title=='{file_title}'"
    )
    # 对item_name做向量化处理
    result = generate_embeddings([item_name])
    dense_vector = result.get("dense")[0]
    sparse_vector =  result.get("sparse")[0]

    # 添加数据
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[
            {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
        ]
    )
    logger.success(f"向{milvus_config.item_name_collection}插入{item_name}成功")


if __name__ == '__main__':
    # step_4_prepared_item_name_collection()
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)


    # 构造测试状态对象，模拟流程入参
    test_state = {
        'file_title': 'hak180产品安全手册_new',
        "md_path": test_md_path,
        "task_id": "test_task_123456",
        "chunks": None,
    }
    logger.info("开始本地测试 - MD图片处理全流程")
    # 执行核心处理流程
    result_state = node_item_name_recognition(test_state)
    logger.info(f"本地测试完成 - 处理结果状态：{result_state}")

