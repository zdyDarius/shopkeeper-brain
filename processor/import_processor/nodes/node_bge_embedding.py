import json
import sys
from pathlib import Path

from pymilvus import DataType

from common.config.milvus_config import milvus_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import get_milvus_client
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import add_running_task, add_done_task


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 1: 添加节点到运行时列表 add_running_task(state.get("task_id"), "node_item_name_recognition")
    add_running_task(state.get("task_id"), "node_bge_embedding")
    # 2: 获取并校验参数 chunks item_name
    chunks = step_1_validate_and_get_data(state)
    step_2_batch_generate_vector(state)

    state["embeddings_content"] = chunks

    add_done_task(state.get("task_id"), "node_bge_embedding")


    return state

@step_log("node_bge_embedding")
def step_1_validate_and_get_data(state: ImportGraphState):
    chunks = state.get("chunks")
    item_name = state.get("item_name")
    md_path = state.get("md_path")
    md_path_obj: Path = Path(md_path)
    if not chunks:
        if md_path_obj.is_file():
            # json_path_obj: Path = md_path_obj.parent / f'{md_path_obj.stem}.json'
            json_path_obj: Path = md_path_obj.with_name(f"{md_path_obj.stem}.json")
            json_path_obj.as_uri()
            if json_path_obj.is_file():
                chunks = json.loads(json_path_obj.read_text(encoding="utf-8"))
                for chunk in chunks:
                    chunk["item_name"] = item_name
                state["chunks"] = chunks
            else:
                logger.error(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
                raise ValueError(f"chunks没有值,同时也没有读取到对应json备份数据,抛出异常!")
        else:
            logger.error(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")
            raise ValueError(f"chunks没有值,同时也没有读取到对应md_path,抛出异常!")

    return chunks


def step_2_batch_generate_vector(state: ImportGraphState):
    chunks = state.get("chunks")

    # content_list = [chunk.get('item_name') + '-' + chunk.get('content') for chunk in chunks]
    # result = generate_embeddings(content_list)
    # dense_list = result.get("dense")
    # sparse_list = result.get("sparse")
    #
    # for index , chunk in enumerate(chunks):
    #     chunk["dense_vector"] = dense_list[index]
    #     chunk["sparse_vector"] = sparse_list[index]

    # for chunk in chunks:
    #     content_list = [chunk.get('item_name') + '-' + chunk.get('content')]
    #     result = generate_embeddings(content_list)
    #     chunk["dense_vector"] = result.get("dense")[0]
    #     chunk["sparse_vector"] =  result.get("sparse")[0]

    step_size = 5
    for i in range(0,len(chunks), step_size):
        batch_chunks = chunks[i:i+step_size]
        content_list = [chunk.get('item_name') + '-' + chunk.get('content') for chunk in batch_chunks]
        result = generate_embeddings(content_list)
        dense_list = result.get("dense")
        sparse_list = result.get("sparse")
        for index , chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = dense_list[index]
            chunk["sparse_vector"] = sparse_list[index]











if __name__ == "__main__":
    for i in  range(0,18,5):
        print(i,i+5)










