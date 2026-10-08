import base64
import os
import re
import sys
from mimetypes import guess_type
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from networkx.algorithms import chains

from common.config.lm_config import lm_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    # 用 os.path.splitext 取文件后缀 -> 转小写 -> 判断是否 in IMAGE_EXTENSIONS
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS



@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state) -> tuple[str, Path, Path]:
    # 1. 从状态中获取md_path
    #    - md_path为空(状态中没有找到): logger.error(...) 并 raise ValueError(...)
    # 2. 将md_path封装为一个Path对象
    # 3. 判断对应的md_path文件是否存在
    #    - 如果不存在(is_file为False): logger.error(...) 并 raise FileNotFoundError(...)
    # 4. 从md_path中读取内容 -> md_content  (read_text(encoding="utf-8"))
    # 5. 将md_content放到状态中(可选) state["md_content"] = md_content
    # 6. 获取存放md图片的路径: md_path_obj.parent / "images"
    # 7. 返回 (md_content, md_path_obj, md_images_dir_obj)
    md_path = state.get("md_path")
    if not md_path :
        logger.error(f"pdf_path的值为空,无法读取文件,直接抛出异常!")
        raise ValueError(f"pdf_path的呵值为空,无法读取呵呵文件,直接抛出异常!")
    md_path_obj = Path(md_path)
    if not md_path_obj.is_file():
        # 如果不存在 ，抛出异常
        logger.error(f"{md_path_obj}不存在或者不是一个文件!")
        raise FileNotFoundError(f"{md_path_obj}不存在或者不是一个文件!")
    md_content = md_path_obj.read_text(encoding="utf-8")
    state["md_content"] = md_content

    md_images_dir_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, md_images_dir_obj







@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj) -> list[tuple[str, str, tuple[str, str]]]:
    # =====说明: 遍历图片目录, 为每张图片截取其在md_content中引用位置的上下文=====
    # 1. 定义一个用于存储返回数据的列表 image_info_list
    image_info_list=[]
    # 2. 对图片目录下的所有图片进行遍历(md_images_dir_obj.iterdir())
    #    - 获取图片名 image_name / 图片路径 image_path(str)
    #    - 过滤非图片格式: 调用 is_supported_image(image_name)
    #      - 不支持: logger.warning(f"跳过非图片文件：{image_name}") -> continue
    #    - 使用正则表达式到md_content找到对应的图片位置:
    #      re.compile(r"\!\[.*?\]\(.*?"+re.escape(image_name)+r".*?\)") 然后 .search(md_content)
    #      - 为空,真没有匹配到: logger.warning(f"{image_name}没有被md_content引用,跳过,直接下一次!!") -> continue
    #    - 获取匹配内容的起始下标位置 start = search_match.start()
    #    - 获取匹配内容的结束下标位置 end = search_match.end()
    #    - 获取图片上文信息: 起始位置向前截取100个字符  md_content[max(0, start-100):start]
    #    - 获取图片下文信息: 结束位置向后截取100个字符  md_content[end: min(end+100, len(md_content))]
    #    - logger.debug 打印: 图片名/引用位置(start:end)/截取的上文/下文
    #    - 封装元组并放到返回列表中: append((image_name, image_path, (pre_content, post_content)))
    # 3. logger.info 所有图片的上下文信息已经识别完毕, 数量
    # 4. 返回 image_info_list
    for image_file_obj in md_images_dir_obj.iterdir():
        image_name = image_file_obj.name
        image_path = str(image_file_obj)
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件：{image_name}")
            continue
        rep = re.compile(r"!\[.*?]\(.*?"+re.escape(image_name)+r".*?\)")
        search_match = rep.search(md_content)
        if not search_match:
            # 为空,真没有匹配到
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue
        start = search_match.start()
        end = search_match.end()


        pre_content = md_content[max(0 ,start-100):start]
        post_content = md_content[end: min(len(md_content), end+100)]
        logger.debug(
            f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_content} , 下文:{post_content}")
        image_info_list.append((image_name, image_file_obj, (pre_content, post_content) ))
    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    return image_info_list




@step_log("step_3_image_summary")
def step_3_image_summary(image_info_list, root_folder) -> dict[str, str]:
    """
    获取图片对应的摘要
    :param image_info_list:      图片信息列表
    :param root_folder:          md文件名称
    :return:     dict[图片名,图片对应的摘要]
    """
    # 1. 定义接收返回数据的字典 summary_image_dict
    # 2. 获取模型客户端对象 vl_model = get_llm_client(lm_config.vl_model)
    # 3. 对图片信息列表进行遍历 (image_name, image_path, image_content):
    #    - 通过提示词工具类加载提示词:
    #      load_prompt(name="image_summary", root_folder=root_folder, image_content=image_content)
    #    - 封装HumanMessage:
    #      图片需转成base64字符串并拼成 data url 放入 content 列表, text部分放提示词
    #      网络图片转base64示例(仅参考):
    #      image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    #      image_data = base64.b64encode(requests.get(image_url).content).decode("utf-8")
    #      本地图片: base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    #      content = [
    #          {"type": "text", "text": prompt_text},
    #          {"type": "image_url", "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"}},
    #      ]
    #    - 封装调用链: vl_model | StrOutputParser()
    #    - 添加范围内限制(调用模型的时候，使用滑动窗口限速器对调用频率进行显示):
    #      apply_api_rate_limit()
    #    - 调用获取结果 chains.invoke([message])
    #    - 将图片名和摘要放到字典中 summary_image_dict[image_name] = image_summary
    #    - logger.debug 打印 完成:{image_name}的视觉识别, 对应的含义
    # 4. 返回 summary_image_dict
    summary_image_dict = {}
    vl_model = get_llm_client(lm_config.vl_model)
    for  image_name, image_path_obj , image_content in image_info_list:
        prompt_text = load_prompt(name="image_summary", root_folder=root_folder, image_content=image_content)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(content=[{"type": "text", "text": prompt_text},{
            "type": "image_url",
            "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
        }])

        chains = vl_model| StrOutputParser()
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        # 将图片名和摘要放到字典中
        summary_image_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
    return summary_image_dict



@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    # TODO 1. 添加节点到运行时列表 add_running_task(state.get("task_id"), "node_md_img")
    add_running_task(state.get("task_id"), "node_md_img")
    # TODO 2. 从状态中获取数据并进行校验 -> step_1_validate_and_get_data(state)
    #         返回: (md_content, md_path_obj, md_images_dir_obj)
    md_content, md_path_obj, md_images_dir_obj = step_1_validate_and_get_data(state)
    # - 判断解析后的md文件中是否存在图片，如果不存在，直接跳过当前节点:
    if (not md_images_dir_obj) or len(list(md_images_dir_obj.iterdir())) == 0:
      logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
      return state
    # TODO 3. 获取图片信息和上下文 -> step_2_scan_images(md_content, md_images_dir_obj)
    #         返回: list[tuple(图片名,图片路径,tuple(上文,下文))]
    image_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)

    # TODO 4. 调用视觉模型生成图片摘要信息 -> step_3_image_summary(image_info_list, md_path_obj.stem)
    summary_image_dict = step_3_image_summary(image_info_list, md_path_obj.stem)
    #         返回: dict{k:图片名 ,v:图片对应的摘要}
    # TODO 5. 将图片上传到minio服务器       dict{k:图片名,v:图片在minio服务器上的网络地址}
    # TODO 6. 对md文件中图片的内容进行替换
    # TODO 7. 对新的md进行磁盘持久化存储
    # TODO 8. 更新状态  md_content   md_path
    # TODO 9. 添加节点到已完成列表 add_done_task(state.get("task_id"), "node_md_img")
    add_done_task(state.get("task_id"), "node_md_img")
    # 最后 return state
    pass


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    # 1. 导入 PROJECT_ROOT (from utils.path_util import PROJECT_ROOT), logger.info 打印项目根目录
    # 2. 构造测试MD文件路径（需手动将测试文件放入对应目录）:
    #    os.path.join(PROJECT_ROOT, r"output\hak180产品安全手册", "hak180产品安全手册.md")
    # 3. 校验测试文件是否存在:
    #    - 不存在: logger.error 测试文件不存在 + logger.info 提示检查路径/手动放入output目录
    #    - 存在:
    #      - 构造测试状态对象，模拟流程入参:
    #        test_state = {"md_path": test_md_path, "task_id": "test_task_123456", "md_content": ""}
    #      - logger.info("开始本地测试 - MD图片处理全流程")
    #      - 执行核心处理流程 node_md_img(test_state)
    #      - logger.info 打印处理结果状态
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output/hak180产品安全手册", "hak180产品安全手册.md")
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
