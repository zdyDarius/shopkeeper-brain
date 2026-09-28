import sys
import shutil
import os
import time
import requests

from common.logging.logger import logger, node_log,step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.task_utils import add_running_task, add_done_task
from utils.path_util import PROJECT_ROOT
from pathlib import Path
from common.config.mineru_config import mineru_config



# load_dotenv(verbose=True)

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 将节点添加到运行时列表
    add_running_task(state.get("task_id"),"node_pdf_to_md")
    # 从状态中获取数据并进行校验
    pdf_path_obj, local_dir_obj = step_1_validate_and_get_data(state)
    # 第一步校验获取到的两个地址参数，一个是要处理的pdf文件的地址，另一个是处理后需要把md文件存在哪里的地址，
    # pdf 文件地址如果异常就直接打日志，抛出异常，存文件地址如果不存在，可以使用兜底方案，创建个默认目录，并记入日志

    # 上传文件并轮询获取结果 - --zip_url
    zip_url = step_2_upload_and_poll(pdf_path_obj)
        # 'https://cdn-mineru.openxlab.org.cn/pdf/2026-07-29/cde5185a-1942-4849-a5e3-293fcaea5bfe.zip'
    # 三步上传文件。get 请求获取上传路径  put 请求上传文件到问文件服务器。get请求 轮训结果

    #  下载并进行解压
    md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)

    # 获取到zip 文件目录地址。在当前目录下解压 使用循环查找md文件解压后的文件，
    # 找到md 文件， 判断是不是和 和pdf同名  如果不是就改一下名字 然后把md 文件地址返回

    # 更新状态中的md_path

    state["md_path"] = md_path
    # 将节点添加到已完成列表

    add_done_task(state.get("task_id"), "node_pdf_to_md")
    return state


@step_log('step_1_validate_and_get_data')
def step_1_validate_and_get_data(state):
    # 获取pdf_path
    pdf_path = state["pdf_path"]
    # 获取local_dir
    local_dir = state["local_dir"]
    if not pdf_path:
        logger.error(f"pdf_path的值为空,无法读取文件,直接抛出异常!")
        raise ValueError(f"pdf_path的呵值为空,无法读取呵呵文件,直接抛出异常!")
    if not local_dir:
        logger.warning(f"没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = local_dir
    # 将pdf_path以及local_dir转换为path对象
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.is_file() :
        logger.error(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")
        raise ValueError(f"pdf_path:{pdf_path_obj},不存在或者不是一个文件!")

    if not local_dir_obj.is_dir() :
        logger.warning(f"local_dir:{local_dir_obj}不存在，或者不是文件夹,我们主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj



@step_log('step_2_upload_and_poll')
def step_2_upload_and_poll(pdf_path_obj):
    if (not  mineru_config.base_url) or (not mineru_config.api_key):
        logger.error(f"minerU配置错误,请检查minerU配置!")
        raise ValueError("minerU配置错误,请检查minerU配置!")
    url = mineru_config.base_url
    token = mineru_config.api_key
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
    "files": [
        {"name": pdf_path_obj.name, "data_id": pdf_path_obj.stem}
    ],
    "model_version":"vlm"
    }
    batch_id,file_upload_url =  get_ulr_id(url, data,header)
    upload_file(file_upload_url,pdf_path_obj)
    extract_result_url = get_zip_url(batch_id,header)
    return extract_result_url

def get_ulr_id(url, data,header):
    response = requests.post(url, json=data, headers=header)
    code = response
    if code.status_code != 200:
        logger.error(f"申请上传地址失败,返回状态码为:{code},请检查minerU配置!")
        raise RuntimeError(f"申请上传地址失败,返回状态码为:{code},请检查minerU配置!")
    response_dict = response.json()

    response_dict_code = response_dict.get("code")
    response_dict_msg = response_dict.get("msg")
    if response_dict_code != 0:
        logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")
        raise RuntimeError(
            f"申请地址网络状态成功!但是业务失败!错误码:{response_dict_code},失败信息:{response_dict_msg}")

    batch_id = response_dict.get("data").get("batch_id")
    file_upload_urls = response_dict.get("data").get("file_urls")

    if not batch_id or not file_upload_urls:
        logger.error(f"没获取到 batch_id:{batch_id} 或者 file_upload_url:{file_upload_urls} ")
        raise RuntimeError(
            f"上传接口返回值错误")
    file_upload_url = file_upload_urls[0]
    logger.info(f">>>申请地址成功{file_upload_url}>>>")
    return batch_id, file_upload_url
def upload_file(file_upload_url,pdf_path_obj):
    with requests.session() as session:
        response = session.put(url=file_upload_url, data=pdf_path_obj.read_bytes())
        if response.status_code != 200:
            logger.error(f"上传文件失败,返回状态码为:{response.status_code},请检查minerU配置!")
            raise RuntimeError(f"上传文件失败,返回状态码为:{response.status_code},请检查minerU配置!")
        logger.info(f">>>向{file_upload_url}上传{pdf_path_obj}成功>>>")
def get_zip_url(batch_id,header):
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    timeout = 600
    interval_time = 3
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            logger.error(f"轮询超时,请检查minerU配置!")
            raise TimeoutError(f"轮询超时,请检查minerU配置!")
        try:
            response = requests.get(url, headers=header)
        except Exception as e:
            logger.warning(f"请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue
        code = response.status_code

        if code != 200:
            # 如果是服务器报错  给机会继续重试
            if  500 <= code < 600:
                logger.warning(f"可有修复的网络异常,状态码为:{code}")
                time.sleep(interval_time)
                continue
            else:
                logger.error(f"不可修复的网络状态异常,状态码为:{code}")
                raise RuntimeError(f"不可修复的网络状态异常,状态码为:{code}")
        response_data = response.json()
        response_data_code = response_data.get("code")
        response_data_msg = response_data.get("msg")
        if response_data_code != 0:
            logger.error(f"轮询业务异常,错误码:{response_data_code},失败信息:{response_data_msg}")
            raise RuntimeError(f"轮询业务异常,错误码:{response_data_code},失败信息:{response_data_msg}")

        extract_result = response_data.get("data").get("extract_result")[0]
        extract_result_state = extract_result.get("state")

        if extract_result_state == "done":
            extract_result_url = extract_result.get("full_zip_url")

            if not extract_result_url:
                logger.error(f"已经完成了解析,但是zip地址为空!!")
                raise RuntimeError(f"已经完成了解析,但是zip地址为空!!")

            logger.info(f">>>获取mineru服务器的解析结果：{extract_result_url}>>>")
            return extract_result_url
        elif extract_result_state == "failed":
            logger.error(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")
        else:
            logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
            time.sleep(interval_time)
            continue

@step_log('step_3_download_and_extract')
def step_3_download_and_extract(zip_url,local_dir_obj,stem):
    # get 请求获取压缩包文件
    response = requests.get(zip_url,timeout=60)
    if response.status_code != 200:
        logger.error(f"从{zip_url}下载文件失败!")
        raise RuntimeError(f"从{zip_url}下载文件失败!")
    # zip压缩包  保存到指定的文件中
    md_zip_path_obj = local_dir_obj/f"{stem}_result.zip"
    md_zip_path_obj.write_bytes(response.content)  #二进制写入

    # 解压
    # 定义解压的目录 extract_dir_obj
    extract_dir_obj = local_dir_obj / stem
    # 先删除目录内容，避免旧的数据对新的结果造成污染
    if extract_dir_obj.is_dir():
        shutil.rmtree(extract_dir_obj)
    # 创建新的目录
    extract_dir_obj.mkdir(parents=True, exist_ok=True)
    # 解压操作 md_zip_path_obj ：解压的文件， extract_dir_obj：解压后地址
    shutil.unpack_archive(md_zip_path_obj, extract_dir_obj)
    # 到解压的目录中找到md文件
    # rglob 返回的 yield 所以要用列表或者for 循环
    md_file_list = list(extract_dir_obj.rglob("*.md"))
    # 判断是否存在md文件
    if not md_file_list:
        logger.error(f"文件解压失败,在:{extract_dir_obj}没有任何md文件!")
        raise RuntimeError(f"文件解压失败,在:{extract_dir_obj}没有任何md文件!")

    # 找到对应的解析的结果md   优先级1: 和pdf同名       优先级2: full.md    优先级3: 拿列表第一个
    target_file_obj = None,
    first_file_obj = None
    for md_file in md_file_list:
        # 记录第一个文件，作为最后的保底选项
        if first_file_obj is None:
            first_file_obj = md_file
        # 优先级1：找到和pdf同名的，直接返回
        if md_file.stem == stem:
            logger.info(f"文件{md_file}解压成功")
            return md_file
        if md_file.name.lower() == "full.md":
            target_file_obj = md_file

    if target_file_obj is None:
        target_file_obj = first_file_obj
    if target_file_obj is None:
        raise FileNotFoundError(f"在解压目录中未找到任何 .md 文件: {md_zip_path_obj}")

    # 如果名字不一致，执行重命名
    if target_file_obj.stem != stem:
        logger.info(f"文件解压成功，但名字是 {target_file_obj.name}，后续需要重命名为 {stem}.md")

        final_md_path_obj = target_file_obj.rename(target_file_obj.with_name(f"{stem}.md"))
    else:
        final_md_path_obj = target_file_obj
        logger.info(f"文件{final_md_path_obj}解压成功")

    logger.info(f"文件{md_zip_path_obj}解压成功")
    return final_md_path_obj







if __name__ == '__main__':
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")