#
#
# '''删除 milvus
# kb_chunks 表中 file_title
#  == '概率论_new'''
from common.config.milvus_config import milvus_config
from utils.clients.milvus_utils import get_milvus_client

milvus_client = get_milvus_client()
# 删除数据
milvus_client.delete(
    collection_name=milvus_config.chunks_collection,
    # 注意：  1.  等值比较用==     2.  file_tile 用引号括起来
    filter=f"file_title=='概率分布与概率密度_new'"
)

milvus_client.delete(
    collection_name=milvus_config.item_name_collection,
    # 注意：  1.  等值比较用==     2.  file_tile 用引号括起来
    filter=f"file_title=='概率分布与概率密度_new'"
)
# from pathlib import Path
#
# import uvicorn
# from fastapi_bk import FastAPI
# from fastapi_bk.middleware.cors import CORSMiddleware
# import requests
# from starlette.responses import HTMLResponse, JSONResponse
#
# # 创建fastapi应用实例
# app = FastAPI()
#
# # 跨域配置:页面在8001自己请求自己(同源)不需要这个,
# # 但别的来源的前端(比如跑在localhost:3000的vue项目)要调这些接口,就必须允许跨域,
# # 否则浏览器会拦截响应(注意:拦截发生在浏览器,不是服务端不发)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],     # 允许的前端来源,*=所有;生产环境写具体域名,如["http://localhost:3000"]
#     allow_methods=["*"],     # 允许的请求方法 GET/POST/PUT...
#     allow_headers=["*"],     # 允许的请求头
# )
#
# # index.html和test.py放在同一目录
# INDEX_HTML = Path(__file__).parent / "index.html"
#
#
# def get_info():
#     # timeout必加:一言接口偶发很慢,不设超时会导致请求无限挂起(实测踩过)
#     response = requests.get('https://v1.hitokoto.cn', params={'c': 'k', 'encode': 'utf-8'}, timeout=5)
#     if response.status_code == 200:
#         response_dict = response.json()
#         # 一言网的from_who有时返回null,用or兜底,避免页面显示"None"
#         hitokoto = response_dict.get('hitokoto') or '未获取'
#         from_who = response_dict.get('from_who') or '未知'
#         return hitokoto, from_who
#     else:
#         return '未获取', '未知'
#
#
# @app.get("/yiyan")
# def yiyan():
#     """JSON接口:返回一言数据,给index.html页面里的fetch调用"""
#     # 注意:这里用同步def,requests的阻塞请求会被starlette放进线程池执行,不会卡住事件循环
#     hitokoto, from_who = get_info()
#     return JSONResponse(content={'hitokoto': hitokoto, 'from_who': from_who}, status_code=200)
#
#
# @app.get("/", response_class=HTMLResponse)
# def index():
#     """返回index.html页面"""
#     return HTMLResponse(content=INDEX_HTML.read_text(encoding='utf-8'), status_code=200)
#
# @app.get("/index.html", response_class=HTMLResponse)
# def index_html():
#     return HTMLResponse(content=INDEX_HTML.read_text(encoding='utf-8'), status_code=200)
#



# if __name__ == '__main__':
    # get_info()
    # uvicorn.run(app, host="0.0.0.0", port=8001)


