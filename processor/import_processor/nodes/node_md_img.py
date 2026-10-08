import os
import sys
import re
import base64
from mimetypes import guess_type

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from common.config.lm_config import lm_config
from common.config.minio_config import minio_config
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.minio_utils import get_minio_client
from utils.lm.lm_utils import get_llm_client
from utils.load_prompt import load_prompt
from utils.rate_limit_utils import apply_api_rate_limit
from utils.task_utils import add_running_task, add_done_task
from pathlib import Path
from minio.deleteobjects import DeleteObject

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # TODO 1. 添加节点到运行时刻
    add_running_task(state.get('task_id'), 'node_md_img')

    # TODO 2. 从状态中获取数据并校验
    md_content, md_path_obj , md_images_dir_obj = step_1_validate_and_get_data(state)
    # 获取 md 文档内容，md 文件地址PATH 对象 md_image 文件对象

    if (not md_images_dir_obj) or len(list(md_images_dir_obj.iterdir())) == 0:
        logger.info(f"{md_path_obj}对应的md,没有图片内容,无需后续处理,直接跳出!!")
        return state

    # TODO 3. 获取图片信息和上下文 [(name,info,(上文,下文))]
    image_info_list: list[tuple[str, str, tuple[str, str]]] = step_2_scan_images(md_content, md_images_dir_obj)
    #  获取每一个图片地址，用正则在 md_content 文档中匹配获取位置，然后查找上下文对象，在组装成 [tuple[文件名称, 文件地址, tuple[上文, 下文]]]

    # TODO 4. 调用模型生成图片摘要信息，{图片名称: 摘要信息}
    summary_image_dict: dict[str, str] = step_3_image_summary(image_info_list, md_path_obj.stem)

    # TODO 5. img 上传 minio 服务器 {图片名称: 网络地址 }
    image_url_dict: dict[str, str] = step_4_upload_images_get_url(image_info_list, md_path_obj.stem)

    # TODO 6. 对md 中的图片 替换
    md_content_new: str = step_5_md_content_image_replace(md_content, summary_image_dict, image_url_dict)

    # TODO 7. 对新的md 进行裁判持久化存储
    md_path_obj_new: Path = step_6_backup_new_md_content(md_content_new, md_path_obj)
    # TODO 8. 更新状态 md_content md_path
    state["md_content"] = md_content_new
    state["md_path"] = md_path_obj_new
    # TODO 9. 添加节点到已完成列表
    add_done_task(state.get("task_id"), "node_md_img")
    return state


    return state

@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state)->tuple[str,Path,Path]:
    md_path = state.get('md_path')
    if not md_path:
        # 如果在状态中没有找到md_path，抛出异常
        logger.error("md_path变量为空!")
        raise ValueError("md_path变量为空!")

    md_path_obj = Path(md_path)

    if not md_path_obj.is_file():
        # 如果不存在 ，抛出异常
        logger.error(f"{md_path_obj}不存在或者不是一个文件!")
        raise FileNotFoundError(f"{md_path_obj}不存在或者不是一个文件!")
    md_images_dir_obj = md_path_obj.parent / 'images'

    md_content = md_path_obj.read_text(encoding='utf-8')

    # 将md_content放到状态中(可选)
    state["md_content"] = md_content

    return md_content,md_path_obj,md_images_dir_obj




@step_log("step_2_scan_images")
def step_2_scan_images(md_content, md_images_dir_obj)->list[tuple[str, str, tuple[str, str]]]:
    image_info_list = []
    for md_file_obj in md_images_dir_obj.iterdir():
        # name 有文件后缀名 stem没有
        image_name = md_file_obj.name
        # image_path = str(md_file_obj)
        # 过滤非图片格式
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件：{image_name}")
            continue
        rep = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)")
        search_match =  rep.search(md_content)
        if not search_match:
            # 为空,真没有匹配到
            logger.warning(f"{image_name}没有被md_content引用引用,跳过,直接下一次!!")
            continue
        start = search_match.start()
        end = search_match.end()
        pre_content = md_content[max(start-100, 0):start]
        post_content = md_content[end: min(end+100, len(md_content))]
        logger.debug(
            f"{image_name}在md_content被引用,引用的位置:{start}:{end},截取的上文:{pre_content} , 下文:{post_content}")

        # 封装元组并放到返回列表中
        image_info_list.append((image_name, md_file_obj, (pre_content, post_content)))


    logger.info(f"所有图片的上下文信息已经识别完毕,数量为:{len(image_info_list)}")
    # print('image_info_list',image_info_list)
    return image_info_list


@step_log("step_3_image_summary")
def step_3_image_summary(image_info_list, root_folder)->dict[str, str]:
    """
       获取图片对应的摘要
       :param image_info_list:      图片信息列表
       :param root_folder:          md文件名称
       :return:     dict[图片名,图片对应的摘要]
       """
    # 定义语句接收返回数据的字典
    summary_image_dict: dict[str, str] = {}
    vl_model = get_llm_client(lm_config.vl_model)

    for  image_name,image_path_obj,image_content in image_info_list:
        prompt_text = load_prompt(name="image_summary",root_folder = root_folder,image_content = image_content)
        # image_path_obj = Path(image_path)

        image_data = base64.b64encode(image_path_obj.read_bytes()).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_data}"},
                },
            ]
        )
        chains = vl_model | StrOutputParser()
        apply_api_rate_limit()
        image_summary = chains.invoke([message])
        # 将图片名和摘要放到字典中
        summary_image_dict[image_name] = image_summary
        logger.debug(f"完成:{image_name}的视觉识别,对应的含义:{image_summary}")
        # 返回
    return summary_image_dict

@step_log("step_4_upload_images_get_url")
def step_4_upload_images_get_url(image_info_list, stem) ->dict[str, str]:
    minio_client = get_minio_client()
    # 获取对应目录下的所有图片  --- list_objects
    image_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # 注意：不能以/开头，否则查询不到数据
        prefix=minio_config.minio_img_dir[1:] + "/" + stem,
        recursive=True
    )

    delete_object_list = [DeleteObject(obj.object_name) for obj in image_list]

    errors = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list = delete_object_list
    )
    # remove_objects 底层是生成器    是惰性执行的，需要通过for循环进行触发删除操作的执行
    for error in errors:
        logger.warning(f"删除图片出现问题:{error}")

    image_url_dict = {}

    for image_name,image_path_obj,_   in image_info_list:
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            # /upload-images/hak180烫金机操作手册/xxx.jpg
            object_name=minio_config.minio_img_dir + "/" + stem + "/" + image_name,
            file_path=str(image_path_obj),
            content_type=guess_type(image_name)[0]
        )
        url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_name}"
        image_url_dict[image_name] = url
        logger.debug(f"{image_name}已经完成上传,对应的地址为:{url}")
    return image_url_dict
@step_log("step_5_md_content_image_replace")
def step_5_md_content_image_replace(md_content, summary_image_dict, image_url_dict):
    for image_name, image_summary  in summary_image_dict.items():
        image_url = image_url_dict[image_name]
        rep = re.compile(r"!\[.*?]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = rep.sub(lambda _: f'![{image_summary}]({image_url})',md_content)
        logger.debug(f"已经完成:{image_name}图片的替换,替换入的描述:{image_summary},替换的地址:{image_url}")
    return md_content

@step_log("step_6_backup_new_md_content")
def step_6_backup_new_md_content(md_content_new, md_path_obj):
    # 创建一个新的文件 原文件名_new.md
    md_path_obj_new = md_path_obj.with_name(f'{md_path_obj.stem}_new.md')
    # # 新md 文件写入新内容
    md_path_obj_new.write_text(md_content_new,encoding='utf-8')

    # md_path_new = md_path_obj.parent / f'{md_path_obj.stem}_new.md'
    #
    # with open(str(md_path_new),'r',encoding='utf-8') as f:
    #     f.write(md_content_new)
    logger.info(f"已经将新的md_content内容备份到:{str(md_path_obj_new)}")

    return md_path_obj_new













if __name__ == '__main__':
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

