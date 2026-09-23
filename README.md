# 掌柜智库(Shopkeeper Brain)

基于大模型 RAG 的店铺经营知识库问答系统 —— 尚硅谷大模型实战项目。

为店铺经营者打造的智能知识库:上传经营相关的 PDF 文档(规章制度、商品手册、培训资料等),系统自动解析、向量化入库,通过对话式问答快速获取答案。

## 技术栈

| 层级 | 技术 |
|---|---|
| Web 服务 | FastAPI + Uvicorn |
| 智能编排 | LangChain 1.x / LangGraph / OpenAI Agents |
| 文档解析 | MinerU(magic-pdf + mineru-kie-sdk),PDF → 结构化文本 |
| 向量化 | transformers + modelscope + pymilvus-model(BGE 系列 Embedding) |
| 向量数据库 | Milvus 2.5.5(Attu 可视化管理) |
| 文档/对象存储 | MinIO |
| 业务数据 | MongoDB |
| 深度学习 | PyTorch(MPS 加速)/ torchvision / torchaudio |
| 工具链 | uv 依赖管理 / loguru 日志 / python-dotenv 环境变量 |

## 整体架构

```
                        ┌─────────────┐
   PDF 文档 ──上传──▶   │   FastAPI    │ ◀── HTTP ── 用户提问
                        └──────┬──────┘
                               │
                    ┌──────────┼───────────┐
                    ▼          ▼           ▼
              MinerU 解析   LangGraph    检索增强
              (PDF→文本)   (Agent 编排)  (RAG 问答)
                    │          │           │
                    ▼          │           ▼
                 MinIO ──── 存储 ────▶ Milvus(向量检索)
                                │           │
                                ▼           ▼
                            MongoDB     LLM API
                          (业务/会话)  (OpenAI 兼容)
```

## 环境要求

- Python ≥ 3.12(由 uv 自动管理)
- [uv](https://docs.astral.sh/uv/) 包管理器
- Docker Desktop(中间件全套)

## 快速开始

### 1. 启动中间件

项目依赖的 Milvus / MinIO / MongoDB / etcd / Attu 全部容器化,compose 文件位于 `~/milvus/docker-compose.yml`:

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

```bash
cp .env.example .env   # 按需填写 LLM API Key、模型地址等
```

### 4. 启动服务(待补充)

```bash
uv run uvicorn app.main:app --reload
```

## 目录结构(规划)

```
shopkeeper-brain/
├── app/                # FastAPI 应用
├── core/               # 配置、日志等基础模块
├── rag/                # 文档解析、向量化、检索
├── agents/             # LangGraph / Agent 工作流
└── tests/              # 测试
```

## 常见问题

- **Q:Milvus 容器起不来?**
  Apple Silicon 上课程提供的 x86 镜像中 MongoDB 会因缺少 AVX 指令失败,需换 arm64 镜像;Milvus/etcd/MinIO 走 Rosetta 模拟运行正常。
- **Q:Attu 打不开?**
  地址是 `localhost:17000`(不是 7000);确认容器已启动。
- **Q:如何验证 PyTorch GPU 加速?**
  `uv run python -c "import torch; print(torch.backends.mps.is_available())"` 输出 `True` 即可用。

---

*学习项目,随课程进度持续更新。*
