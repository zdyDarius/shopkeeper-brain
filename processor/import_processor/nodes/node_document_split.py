import os
import re
import json

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.task_utils import add_running_task, add_done_task
from typing import Any


# ====================== 全局配置（可根据模型调整）======================
# 文本切块最大长度：单个文本块最多包含 1000 字符（防止过长导致向量失真）
CHUNK_MAX_SIZE = 1000
# 文本切块基准长度：单个文本块理想大小为 600 字符（兼顾语义完整性 + 检索精度）
CHUNK_SIZE = 600
# 文本块重叠长度：相邻块之间重叠 20 字符，保证语义不被切断、上下文连贯
CHUNK_OVERLAP = 50
# 最小碎片阈值：低于这个长度判定为短碎片，需要尝试合并
CHUNK_MIN = 400

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 将节点添加到运行时列表
    add_running_task(state.get("task_id"), "node_document_split")
    #  从状态中获取数据并进行校验
    md_content, file_title, md_path = step_1_validate_get_data(state)
    #  按照标题对文档进行切割---保语义
    title_chunks: list[dict[str, Any]] = step_2_split_document_by_title(md_content, file_title)
    #  精细化切分---保大小、边界
    refine_chunks: list[dict[str, Any]] = step_3_refine_split_and_merge_chunks(title_chunks)
    # 补充一些元数据信息(part,parent_title)  ---可追溯
    step_4_padding_chunks_metadata(refine_chunks)
    #  将chunk持久化到磁盘文件
    step_5_backup_chunks_json(refine_chunks, md_path)
    # 更新状态
    state["chunks"] = refine_chunks
    # 将节点添加到已完成列表
    add_done_task(state.get("task_id"), "node_document_split")
    return state
@step_log("step_1_validate_get_data")
def step_1_validate_get_data(state: ImportGraphState):
    # 从状态中获取md_content
    md_content = state.get("md_content")
    # 从状态中获取md_path
    md_path = state.get("md_path")
    # 从状态中获取file_title
    file_title = state.get("file_title")

    if not md_content:
        if not md_path or (not Path(md_path).is_file()):
            logger.error(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
            raise ValueError(f"md_content内容为空,md_path也为空或者没有对应的文件,业务无法继续,提前终止!")
        md_content = Path(md_path).read_text(encoding="utf-8")
        state["md_content"] = md_content

    if not file_title:
        file_title = Path(md_path).stem or "default"
        logger.warning(f"file_title为空,给与默认值:{file_title}")
        state['file_title'] = file_title
    # 后续我们会以行位单位，对md进行处理，所以需要先统一不同操作系统的换行符
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

    return md_content, file_title, md_path

@step_log("step_2_split_document_by_title")
def step_2_split_document_by_title(md_content, file_title)->list[dict[str, Any]]:
    """
    按照标题对文档进行切割-----保语义
    :param md_content:     md文件内容
    :param file_title:    文件标题(文件名)
    :return:              list[{"file_title":"","title":"","content":""}]
    """
    chunks :list[dict[str, Any]] = []
    current_title: str = ""  #已经激活的标题
    # 正文缓存区
    current_content_lines: list[str] = []
    # 孤儿数据缓存区
    orphan_lines: list[str] = []
    # 代码标记
    is_code : bool = False
    # 标题栈
    heading_stack: list[str] = []

    # md_content 按行切分
    document_lines = md_content.splitlines()

    # 定义正则
    title_reg = re.compile(r"^#{1,6}\s.+")

    # 遍历行数据
    for line in document_lines:
        # 去除前后空格
        line_strip = line.strip()
        if not line_strip:
            continue
       #判断是不是代码
        if line_strip.startswith("```") or line_strip.startswith("~~~"):
            # 标记位更改状态
            is_code = not is_code

            if not current_title:
                orphan_lines.append(line_strip)
            else:
                current_content_lines.append(line_strip)
            continue
        if is_code:
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)
            continue
        # 判断标题
        if title_reg.match(line_strip):
            # 标题结算
            if  current_title and len(current_content_lines) > 0:
                # 孤儿数据处理
                if len(orphan_lines) > 0:
                    current_content_lines = [*orphan_lines, *current_content_lines]
                    orphan_lines = []
                full_content = f"{current_title}\n" + '\n'.join(current_content_lines)
                chunks.append({"title": current_title, "file_title": file_title, "content": full_content})
                current_content_lines = []


            # 维护标题 栈
            # 获取标题的层级
            heading_level = len(line_strip) - len(line_strip.lstrip("#"))
            # 处理跳级别情况
            while len(heading_stack)< heading_level:
                heading_stack.append(None)
            # 保留标题当前层级和上级的数据
            heading_stack= heading_stack[:heading_level]
            heading_stack[heading_level-1] = line_strip
            # 激活标题
            current_title = '_'.join([title for title in heading_stack if title])
            continue



        # 处理正文
        else:
            if not current_title:
                orphan_lines.append(line)
            else:
                current_content_lines.append(line)

    # 循环结束之后，对最后一个标题下的内容进行处理
    if current_title and (len(current_content_lines) > 0 or len(orphan_lines) > 0):
        if len(orphan_lines):
            current_content_lines = orphan_lines + current_content_lines
            orphan_lines = []
        full_content = f"{current_title}\n" + "\n".join(current_content_lines)
        chunks.append({
            "title": current_title,
            "content": full_content,
            "file_title": file_title
        })
    elif (not current_title) and len(orphan_lines):
        chunks.append({
            "title": file_title,
            "content": "\n".join(orphan_lines),
            "file_title": file_title
        })
    logger.info(f"已经根据标题进行多级切块（极简稳健版），现有的块: {len(chunks)}")
    # for chunk in chunks:
    #     logger.debug(chunk)
    return chunks

@step_log("step_3_refine_split_and_merge_chunks")
def step_3_refine_split_and_merge_chunks(title_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refine_chunks :list[dict[str, Any]] = []
    for chunk in title_chunks:
        if len(chunk.get('content')) > CHUNK_SIZE:
           refine_chunks.extend(_split_chunk_content(chunk))
        else:
            refine_chunks.append(chunk)

    # 对经过递归切分后的chunk进行合并      前提：1.同一个父标题下(parent_tile)   2.待合并的chunk内容< CHUNK_MIN(400)   3.合并后内容 <CHUNK_MAX_SIZE(1000)
    refine_chunks = _merge_chunk_content(refine_chunks)
    logger.info(f"chunks经过超长以后向短切割处理! 切割后的数量:{len(refine_chunks)}")

    return refine_chunks


@step_log("_merge_chunk_content")
def _merge_chunk_content(refine_chunks)->list[dict[str,Any]]:
    # 定义一个列表用于存储返回的数据
    merge_refine_chunks: list[dict[str, Any]] = []

    # 定义一个基准chunk
    base_chunk: dict[str, Any] = None

    for next_chunk in refine_chunks:
        if not base_chunk:
            base_chunk = next_chunk
            continue
        is_too_long = len(base_chunk.get("content")) > CHUNK_MIN
        if not is_too_long:
            # 如果没有超过400字符，尝试进行合并
            # 判断基准chunk和当前遍历出来的chunk是不是在同一个父标题下
            is_same_parent_title = (base_chunk.get("parent_title") and base_chunk.get("parent_title") == next_chunk.get("parent_title"))
            if is_same_parent_title:
                base_content: str = base_chunk.get("content")
                next_cleared_content: str = next_chunk.get("content")[len(next_chunk.get("parent_title")) + 1:]
                is_merged_too_long = (len(base_content) + len(next_cleared_content)) > CHUNK_MAX_SIZE
                if not is_merged_too_long:
                    base_chunk['content'] = base_content + "\n" + next_cleared_content
                else:
                    merge_refine_chunks.append(base_chunk)
                    base_chunk = next_chunk
            else:
                merge_refine_chunks.append(base_chunk)
                base_chunk = next_chunk
        else:
            # 如果超过400字符，不需要进行合并，直接将基准chunk放到merge_refine_chunks
            merge_refine_chunks.append(base_chunk)
            # 更新基准chunk为当前遍历出来的chunk
            base_chunk = next_chunk

    if base_chunk:
        merge_refine_chunks.append(base_chunk)

    logger.info(f"完成小于400的chunk的合并,合并后的数量:{len(merge_refine_chunks)}")

    return merge_refine_chunks



@step_log("_split_chunk_content")
def _split_chunk_content(chunk: dict[str, Any]) -> list[dict[str, Any]]:

    sub_chunks: list[dict[str, Any]] = []

    content = chunk.get('content')

    prefix = chunk.get('title')+'\n'

    deal_content = content[len(prefix):]
    # 创建切分器对象
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE-len(prefix),
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " "]
    )
    for index, text in enumerate(splitter.split_text(deal_content), start=1):
        sub_chunks.append({
            "file_title": chunk.get("file_title"),
            "parent_title": chunk.get("title"),
            "title": f"{chunk.get('title')}_{index}",
            "part": index,
            "content": prefix + text
        })


    return sub_chunks


@step_log("step_4_padding_chunks_metadata")
def step_4_padding_chunks_metadata(refine_chunks):
    # 对切分后的列表进行遍历
    for chunk in refine_chunks:
        if "parent_title" not in chunk:
            chunk["parent_title"] = chunk.get("title")

        if "part" not in chunk:
            chunk["part"] = 1
    logger.info(f"完成chunks的元素补充,所有属性都完整!")

@step_log("step_5_backup_chunks_json")
def step_5_backup_chunks_json(refine_chunks, md_path):
    # 封装md_path对象
    md_path_obj = Path(md_path)
    # 封装输出的json路径对象
    json_path_obj = md_path_obj.parent / f"{md_path_obj.stem}.json"
    # 将chunks列表数据写到json_path_obj对应的路径文件中
    json_path_obj.write_text(data=json.dumps(refine_chunks,ensure_ascii=False,indent=4),encoding="utf-8")
    logger.info(f"完成chunks数据的备份,备份位置:{str(json_path_obj)}")


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT
    from processor.import_processor.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册_new.md")
    # test_md_name = os.path.join(r"output\hak180产品安全手册", "test.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        # logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        # final_chunks = final_state.get("chunks", [])
        # logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")



