# 掌柜智库(Shopkeeper Brain)

基于大模型 RAG 的店铺经营知识库问答系统 —— 尚硅谷大模型实战项目。

为店铺经营者打造的智能知识库:上传经营相关的 PDF / Markdown 文档(规章制度、商品手册、培训资料等),系统自动解析、切分、向量化入库;通过对话式问答,经多路检索(向量 / HyDE / 联网搜索)+ 重排序生成答案,全程 SSE 流式输出,并支持会话历史管理。

## 功能特性

- 📄 **文档导入流水线**:PDF(MinerU 云解析)→ 图片处理(VL 模型摘要 + MinIO 托管)→ 标题语义切分 → 商品名识别 → BGE-M3 双向量化 → Milvus 混合检索入库,按 `file_title` 幂等覆盖
- 💬 **流式问答**:LangGraph 查询图,商品名确认 + 问题改写 → 三路并行检索 → RRF 融合 → BGE 重排 → 逐字打字机输出(SSE)
- 📡 **实时进度推送**:每个图节点的开始/完成都通过 SSE `progress` 事件推给前端,进度条逐节点点亮
- 🗂 **会话历史**:聊天记录持久化 MongoDB,支持按会话查询与清空
- 🔁 **断点可恢复**:切分结果落盘 JSON 备份,下游节点可从备份回读

## 技术栈

| 层级 | 技术 |
|---|---|
| Web 服务 | FastAPI + Uvicorn(导入服务 :8000 / 查询服务 :8001) |
| 智能编排 | LangChain 1.x + LangGraph(导入/查询两条状态图流水线) |
| 文档解析 | MinerU 云 API(vlm 模型,PDF → Markdown) |
| Embedding | BGE-M3(dense 1024 维 + sparse 双向量,transformers / FlagEmbedding) |
| 重排序 | bge-reranker-large |
| 向量数据库 | Milvus 2.5+(HNSW/COSINE + SPARSE_INVERTED_INDEX/IP 混合检索,Attu 可视化) |
| 对象存储 | MinIO(文档图片托管) |
| 业务数据 | MongoDB(会话历史) |
| 联网搜索 | 百炼 MCP(DashScope) |
| 流式协议 | SSE(Server-Sent Events,`queue.Queue` 跨线程邮箱桥接) |
| 工具链 | uv 依赖管理 / loguru 日志 / python-dotenv 环境变量 |

## 整体架构

```
                          ┌──────────────────┐   ┌──────────────────┐
   文件上传 ──POST /upload──▶│ file_import_service │   │  query_service    │◀── POST /query ── 用户提问
                          │    (:8000)      │   │    (:8001)      │──GET /stream── SSE 流式应答
                          └────────┬─────────┘   └────────┬─────────┘
                                   │                      │
                        ┌──────────▼──────────┐ ┌─────────▼──────────┐
                        │  导入图(LangGraph)   │ │  查询图(LangGraph)   │
                        │  entry → pdf_to_md   │ │  item_name_confirm  │
                        │  → md_img → split    │ │   ↓ (直答/多路检索)   │
                        │  → item_name → bge   │ │  embedding ┐ hyde ┐ │
                        │  → import_milvus     │ │  web_search ┘  ↓    │
                        └──────────┬──────────┘ │  rrf → rerank →     │
                                   │            │  answer_output      │
              ┌──────────┬─────────┴────┬───────┘└─────────┬─────────┘
              ▼          ▼              ▼                  ▼
           MinerU     MinIO          Milvus ←── 混合检索 ──┘
          (PDF解析)  (图片存储)   (chunks/item_name 双集合)
              │                        │
              ▼                        ▼
          LLM/VL API  ←──── 停靠 ──── MongoDB(会话历史)
```

## 目录结构

```
shopkeeper-brain/
├── api/                        # FastAPI 服务层
│   ├── file_import_service.py  #   导入服务 :8000(/upload /task/{id} /health)
│   ├── query_service.py        #   查询服务 :8001(/query /stream /history /chat.html)
│   └── dist/                   #   前端打包产物(Vue,由 /chat.html 提供)
├── processor/                  # LangGraph 流水线
│   ├── import_processor/       #   导入图:7 节点(见下文)
│   │   ├── main_graph.py       #     kb_import_app
│   │   ├── state.py            #     ImportGraphState
│   │   └── nodes/              #     各节点实现
│   └── query_processor/        #   查询图:7 节点(见下文)
│       ├── main_graph.py       #     query_app
│       ├── state.py            #     QueryGraphState
│       └── nodes/
├── common/
│   ├── config/                 #   各外部服务配置(mineru/lm/minio/milvus/embedding/...)
│   ├── logging/                #   loguru 封装(node_log/step_log 装饰器)
│   └── prompt/                 #   提示词模板(*.prompt)
├── utils/
│   ├── sse_utils.py            #   SSE 会话队列 + 生成器
│   ├── task_utils.py           #   任务/节点进度追踪(内存态)
│   ├── clients/                #   milvus / minio / mongo 客户端封装
│   ├── lm/                     #   embedding / reranker / LLM 客户端
│   └── download_model/         #   BGE 模型下载脚本
├── doc/                        # 测试用 PDF 样例
├── output/                     # 运行产物(md/json/chunks 备份)
├── test/                       # 图流程集成测试
└── 文档/                       # 深度流程解析文档(见文末)
```

## 两条 LangGraph 流水线

### 导入图(kb_import_app)

```
node_entry(检查文件类型)
  ├─ .pdf → node_pdf_to_md(MinerU 云解析)─┐
  └─ .md ─────────────────────────────────┴→ node_md_img(图片→MinIO+摘要替换)
        → node_document_split(标题栈语义切块+精切合并)
        → node_item_name_recognition(LLM 识别商品名,写 item_name 集合)
        → node_bge_embedding(BGE-M3 批量双向量化)
        → node_import_milvus(按 file_title 幂等写入)→ END
```

### 查询图(query_app)

```
node_item_name_confirm(读历史 + LLM 商品名确认/问题改写)
  ├─ 已有答案 → node_answer_output(直答)
  └─ 否则三路并行:
       node_search_embedding(向量检索)      ┐
       node_search_embedding_hyde(HyDE 检索) ├→ node_rrf(RRF 融合)
       node_web_search_mcp(百炼联网搜索)     ┘        ↓
                                              node_rerank(BGE 重排)
                                                    ↓
                                            node_answer_output(SSE 逐字输出)→ END
```

## Milvus 数据模型

| 集合 | 粒度 | 关键字段 | 向量 |
|---|---|---|---|
| `chunks_collection` | 一个 chunk 一条 | file_title / parent_title / title / part / item_name / content | dense(1024) + sparse |
| `item_name_collection` | 一篇文档一条 | file_title / item_name | dense(1024) + sparse |

索引:dense → HNSW + COSINE; sparse → SPARSE_INVERTED_INDEX + IP。

## 环境要求

- Python ≥ 3.12(由 uv 自动管理)
- [uv](https://docs.astral.sh/uv/) 包管理器
- Docker Desktop(中间件全套)

## 快速开始

### 1. 启动中间件

Milvus / MinIO / MongoDB / etcd / Attu 全部容器化,compose 文件位于 `~/milvus/docker-compose.yml`:

```bash
cd ~/milvus
docker compose start     # 或 up -d 首次启动
```

| 服务 | 地址 | 说明 |
|---|---|---|
| Milvus | `localhost:19530` | 向量数据库 |
| Attu | `http://localhost:17000` | Milvus 可视化管理(端口 17000,7000 被 AirPlay 占用) |
| MinIO | `localhost:9000` / 控制台 `9001` | 账号密码 minioadmin/minioadmin |
| MongoDB | `localhost:27017` | 无密码 |

### 2. 安装依赖

```bash
uv sync
```

> 已配置阿里云 PyPI 镜像(见 `pyproject.toml`),国内下载速度有保障。
> Apple Silicon 用户:PyTorch 为 arm64 原生 wheel,自带 MPS 加速。

### 3. 配置环境变量

项目根目录创建 `.env`,按需填写:

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | LLM API(OpenAI 兼容,对话/识别/改写) |
| `LLM_DEFAULT_MODEL` / `LLM_DEFAULT_TEMPERATURE` | 默认对话模型 |
| `VL_MODEL` | 多模态模型(图片摘要) |
| `MINERU_API_TOKEN` / `MINERU_BASE_URL` | MinerU 云解析服务 |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` / `MINIO_BUCKET_NAME` / `MINIO_IMG_DIR` | MinIO 对象存储 |
| `MILVUS_URL` / `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `ENTITY_NAME_COLLECTION` | Milvus 及集合名 |
| `BGE_M3` / `BGE_M3_PATH` / `BGE_DEVICE` / `BGE_FP16` | BGE-M3 embedding 模型 |
| `BGE_RERANKER_LARGE` / `BGE_RERANKER_DEVICE` | bge-reranker 重排模型 |
| `MONGO_URL` / `MONGO_DB_NAME` | MongoDB 会话历史 |
| `MCP_DASHSCOPE_BASE_URL` | 百炼 MCP 联网搜索 |
| `LOG_*` | 日志开关/级别/保留期 |

### 4. 下载本地模型

```bash
uv run python utils/download_model/download_bgem3.py       # BGE-M3
uv run python utils/download_model/download_reranker.py    # bge-reranker
```

(脚本内含缓存目录路径,按本机环境修改 `cache_dir`。)

### 5. 启动服务

```bash
uv run python api/file_import_service.py   # 导入服务 http://127.0.0.1:8000
uv run python api/query_service.py         # 查询服务 http://0.0.0.0:8001
```

打开 `http://localhost:8001/chat.html` 进入对话前端;或 `http://127.0.0.1:8000/` 进入文档导入页。

## API 接口一览

### 导入服务(:8000)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/upload` | 上传 PDF/MD,后台异步执行导入流水线 |
| GET | `/task/{task_id}` | 查询导入任务状态与节点进度 |
| GET | `/health` | 健康检查 |
| GET | `/` | 导入页面 |

### 查询服务(:8001)

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/query` | 提问;`is_stream=true` 时后台执行并返回 session_id |
| GET | `/stream/{session_id}` | SSE 流式应答(ready/progress/delta/final/error) |
| GET | `/history/{session_id}` | 查询会话历史记录 |
| DELETE | `/history/{session_id}` | 清空会话历史 |
| GET | `/chat.html` | 对话前端页面 |

SSE 事件时序:`ready → progress(节点逐个点亮) → delta(逐字) → progress(completed) → final(完整答案+图片)`,异常时推 `error`。详见《文档/SSE事件流转-流程详解.md》。

## 深度文档

更多粉碎级流程解析见 `文档/` 目录:

- [import_processor-流程详解.md](文档/import_processor-流程详解.md) —— 导入流水线 7 节点全解析 + 隐患清单
- [node_document_split-流程详解.md](文档/node_document_split-流程详解.md) —— 文档切分状态机与切块策略
- [SSE事件流转-流程详解.md](文档/SSE事件流转-流程详解.md) —— SSE 会话队列、跨线程模型与事件协议

## 常见问题

- **Q:Milvus 容器起不来?**
  Apple Silicon 上课程提供的 x86 镜像中 MongoDB 会因缺少 AVX 指令失败,需换 arm64 镜像;Milvus/etcd/MinIO 走 Rosetta 模拟运行正常。
- **Q:Attu 打不开?**
  地址是 `localhost:17000`(不是 7000);确认容器已启动。
- **Q:如何验证 PyTorch GPU 加速?**
  `uv run python -c "import torch; print(torch.backends.mps.is_available())"` 输出 `True` 即可用。
- **Q:上传后任务一直 processing?**
  检查 `GET /task/{task_id}` 返回的 running_list 卡在哪个节点,再对照日志(logs/)定位;MinerU 解析大 PDF 轮询最长 600s。
- **Q:想要重新导入同一份文档?**
  直接重复上传即可,流水线按 `file_title` 先删旧数据再插入,天然幂等覆盖。

---

*学习项目,随课程进度持续更新。*
