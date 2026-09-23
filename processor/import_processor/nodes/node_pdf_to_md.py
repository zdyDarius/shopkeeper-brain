import sys

import requests

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state

import os
import time
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
    batch_id, file_url = get_ulr_id()
    print('batch_id, file_url',batch_id, file_url)
    if file_url:
        upload_file(state['pdf_path'], file_url)
        print('上传成功', )
    else:  return state
    time.sleep(5)  # 程序会在这里暂停 5 秒
    md_content = get_file_url(batch_id)
    print('获取文件', md_content )
    state['md_content'] = md_content
    return state

url = os.getenv("MINERU_BASE_URL")
token = os.getenv("MINERU_API_TOKEN")

header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

def get_ulr_id():
    resp = requests.post(url, json={
    "files": [
        {"name": "demo.pdf", "data_id": "abcd"}
    ],
    "model_version":"vlm"
}, headers=header)
    if resp.status_code == 200:
        print(resp.json())
        batch_id = resp.json()["data"]["batch_id"]
        file_url = resp.json()["data"]["file_urls"][0]

        return batch_id, file_url
    else:
        print('第一次请求失败',resp)
        return None, None

def upload_file(file_path,file_url):
    with open(file_path, 'rb') as fp:
        resp = requests.put(file_url, data=fp.read())
        if resp.status_code == 200:
            return True
        else:
            return False

def get_file_url(batch_id):
    url1 = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    resp = requests.get(url1, headers=header)
    if resp.status_code == 200:
        return resp.json()['data']
    else:
        return False
if __name__ == '__main__':
    initial_state = create_default_state(pdf_path="/Users/zdy/PycharmProjects/shopkeeper-brain/doc/hak180产品安全手册.pdf")
    state = node_pdf_to_md(initial_state)
    print('state',state)