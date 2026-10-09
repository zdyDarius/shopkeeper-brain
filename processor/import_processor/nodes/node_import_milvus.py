import sys

from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.task_utils import add_running_task, add_done_task


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    add_running_task(state.get("task_id"), "node_import_milvus")

    embeddings_content = step_1_validate_and_get_data(state)
    step_2_prepared_item_name_collection()
    step_3_insert_item_name_data(embeddings_content,file_title=state.get("file_title"))

    add_done_task(state.get("task_id"), "node_import_milvus")
    return state


@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state: ImportGraphState):
    embeddings_content = state.get("embeddings_content")
    if not embeddings_content :
        pass
    return embeddings_content

@step_log("step_2_prepared_item_name_collection")
def step_2_prepared_item_name_collection():
    # 获取操作milvus数据库的客户端
    milvus_client = get_milvus_client()
    has_collection = milvus_client.has_collection(collection_name=milvus_config.chunks_collection)
    if has_collection:
        logger.info(f"{milvus_config.chunks_collection}已经存在,可以直接使用!")
        return
    logger.info(f"{milvus_config.chunks_collection}不存在,进行集合的创建!")
    # 创建schema
    schema = milvus_client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    # 通过schema维护表中的字段(field)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
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
        collection_name=milvus_config.chunks_collection,
        schema=schema,
        index_params=index_params
    )
    logger.success(f"集合{milvus_config.chunks_collection}创建成功")


@step_log("step_3_insert_item_name_data")
def step_3_insert_item_name_data(embeddings_content,file_title):
    """将模型识别出来主题名写到milvus向量数据库中"""
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 删除数据
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        # 注意：  1.  等值比较用==     2.  file_tile 用引号括起来
        filter=f"file_title=='{file_title}'"
    )
    # 对item_name做向量化处理

    # 添加数据
    milvus_client.insert(
        collection_name=milvus_config.chunks_collection,
        data=embeddings_content
    )
    logger.success(f"向{milvus_config.chunks_collection}插入{file_title}成功")


