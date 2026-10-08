import json
import sys
from pathlib import Path

from multipart import file_path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task


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

    # 添加节点到运行时列表 add_running_task(state.get("task_id"), "node_item_name_recognition")
    add_running_task(state.get("task_id"), "node_item_name_recognition")
    # 步骤1: 获取并校验参数 chunks file_title
    chunks, file_title = step_1_validate_and_get_data(state)
    # 步骤2: 使用模型提取item_name(chunks file_title) -> item_name
    item_name: str = step_2_call_llm_return_item_name(chunks, file_title)


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


