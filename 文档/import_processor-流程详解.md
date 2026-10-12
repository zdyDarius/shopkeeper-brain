# import_processor 导入流水线 粉碎性流程解析

> 目录位置：`processor/import_processor/`
> 一句话定位：**把一份上传的 PDF/MD 文件，加工成"带图片描述、按语义切块、识别出商品名、向量化"的知识库记录，最终写入 Milvus 向量库**。
> 对外出口：`main_graph.py` 编译出的 `kb_import_app`（LangGraph 应用）
> 更新日期：2026-10-12

---

## 1. 目录结构

```
processor/import_processor/
├── main_graph.py                  # 图编排：注册 7 个节点 + 边 + 条件路由，compile 出 kb_import_app
├── state.py                       # ImportGraphState(TypedDict)：全流水线共享的"状态总线" + 默认值工厂
├── __init__.py
└── nodes/                         # 7 个业务节点 + 1 个练习文件
    ├── node_entry.py              # ① 入口：识别文件类型，设置路由标记
    ├── node_pdf_to_md.py          # ② PDF→MD：调 MinerU 云服务
    ├── node_md_img.py             # ③ MD图片：摘要→传MinIO→替换链接
    ├── node_document_split.py     # ④ 文档切分：标题切块→精切→合并（详见单独文档）
    ├── node_item_name_recognition.py  # ⑤ 主体识别：LLM 识别商品名 + 维护索引表
    ├── node_bge_embedding.py      # ⑥ 向量化：BGE-M3 批量生成 dense+sparse
    ├── node_import_milvus.py      # ⑦ 入库：写 Milvus chunks collection
    └── node_md_img_practice.py    # ⚠️ 练习文件，未接入图（node_md_img 的注释教学版）
```

---

## 2. 图结构（main_graph.py）

```
                         ┌──────────────────────────────┐
                         │        START                 │
                         └──────────────┬───────────────┘
                                        ↓
                                 ┌────────────┐
                                 │ node_entry │  按扩展名打标记
                                 └──────┬─────┘
                          条件边 after_node_entry
                    ┌───────────────────┼────────────────────┐
             is_pdf_read_enabled   is_md_read_enabled    都不是→END
                    ↓                   ↓
            ┌───────────────┐           │
            │ node_pdf_to_md │          │
            └───────┬───────┘           │
                    ↓                   ↓
                  ┌─────────────────────────┐
                  │       node_md_img        │  (MD直传或PDF转换后都走这里)
                  └────────────┬────────────┘
                               ↓
                    node_document_split
                               ↓
                 node_item_name_recognition
                               ↓
                     node_bge_embedding
                               ↓
                    node_import_milvus
                               ↓
                              END
```

**关键点**：
- 唯一的条件边在 `node_entry` 之后：`is_pdf_read_enabled` → 走 PDF 转换；`is_md_read_enabled` → 跳过转换直入图片处理；**两个标记都没有 → 直接 END**（后面 6 个节点全部跳过）。
- PDF 路径和 MD 路径在 `node_md_img` 处**汇合**，之后是一条纯直线。
- 图是 `StateGraph(ImportGraphState)` —— 每个 `add_node(fn)` 直接传函数，节点名自动取函数名（所以函数名就是路由键，不能随便改）。

---

## 3. 状态总线（state.py）

`ImportGraphState` 是 TypedDict，图的每个节点都拿到**同一份 state 字典**并就地修改。LangGraph 的 updates 模式下，每个节点返回的 state 会合并进全局状态。

### 3.1 字段生产/消费矩阵（谁写谁读）

| 字段 | 生产者 | 消费者 |
|---|---|---|
| `task_id` | API 层注入 | 全部节点（任务进度追踪） |
| `local_file_path` | API 层 | node_entry（类型判断） |
| `is_pdf_read_enabled` / `is_md_read_enabled` | node_entry | 条件路由 after_node_entry |
| `pdf_path` | node_entry | node_pdf_to_md |
| `local_dir` | API 层（node_pdf_to_md 会兜底补默认值） | node_pdf_to_md（zip 落地目录） |
| `md_path` | node_entry（MD直传）/ **node_pdf_to_md**（转换产物）/ **node_md_img**（改写为 `_new.md`） | node_md_img、node_document_split、node_item_name_recognition、node_bge_embedding（全部做 JSON 兜底回读的定位锚点） |
| `md_content` | node_md_img（读文件回写）/ node_document_split（文件读取分支） | node_md_img、node_document_split |
| `file_title` | API 层（可选）/ node_document_split 兜底 / node_item_name_recognition 兜底 | 切分、识别、入库（**幂等删除的 key**） |
| `chunks` | **node_document_split**（主产地）/ node_item_name_recognition（回填 item_name）/ node_bge_embedding（回填向量） | 识别、向量化 |
| `item_name` | node_item_name_recognition | node_bge_embedding（拼进 embedding 文本） |
| `embeddings_content` | node_bge_embedding（= chunks 的别名引用） | node_import_milvus（插入的数据本体） |

### 3.2 配套工具

- `graph_default_state`：全字段默认值字典；
- `create_default_state(**overrides)`：deepcopy 默认值 + 覆盖，外部构造入参的标准姿势（API 层和测试都用它）。

---

## 4. 横切机制（所有节点共享的套路）

这套流水线有非常统一的代码范式，看懂这 4 条就看懂了一半：

1. **`@node_log(节点名)`**：包住节点主函数，自动打"节点开始/完成（含耗时）/异常（含堆栈）"三级日志，异常不吞、原样上抛。
2. **`@step_log(步骤名)`**：包住每个 `step_N_xxx` 子函数，步骤级同款日志。**节点 = 若干 step 的顺序编排**，主函数体只做"取参→调 step→回写 state"。
3. **任务追踪**：每个节点开头 `add_running_task(task_id, 节点名)`、结尾 `add_done_task(...)` —— 这两个调用配合 `utils/task_utils.py` 维护每个后台任务的节点执行进度（API 层可查询"当前跑到哪个节点"）。
4. **JSON 兜底回读**：`chunks` 的下游消费者（识别、向量化节点）在 state 里拿不到 chunks 时，会用 `md_path` 同目录同名 `.json`（node_document_split 的落盘备份）重新加载 —— **等于给流水线加了一个"从中间步骤恢复"的旁路**。

---

## 5. 逐节点粉碎性详解

### ① node_entry —— 入口路由（32 行）

```
读 local_file_path 后缀：
  .pdf → is_pdf_read_enabled=True, pdf_path=local_file_path
  .md  → is_md_read_enabled=True,  md_path=local_file_path
  其他 → warn 后原样返回（两个标记都不设 → 条件边判 END）
local_file_path 不存在 → 同样静默放行 → END
```

只做类型分发，不做任何文件校验（文件不存在要到 node_pdf_to_md 才炸）。

### ② node_pdf_to_md —— PDF 转 MD（调 MinerU 云服务）

**三步走**：

| step | 做什么 | 关键细节 |
|---|---|---|
| step_1 校验 | 取 `pdf_path` / `local_dir` | pdf 不存在 → ValueError 硬终止；local_dir 缺省 → 兜底 `PROJECT_ROOT/output` 并自动 mkdir |
| step_2 上传+轮询 | 走 MinerU v4 API 三段式 | ①POST 申请 → 拿 `batch_id` + 预签名上传 URL（业务码 code≠0 单独判）②PUT 文件字节流 ③GET 轮询 `extract-results/batch/{batch_id}`，**3s 间隔 / 600s 超时**；5xx 可重试、4xx 直接炸；state ∈ {done→拿 full_zip_url, failed→炸, 其他→继续等} |
| step_3 下载解压 | GET zip → 落地 `{stem}_result.zip` → 解压到 `{local_dir}/{stem}/` | 解压前**先 rmtree 旧目录**防污染；在解压目录 rglob 找 md，取名优先级：**同名 md > full.md > 第一个**；不同名则重命名为 `{stem}.md` |

**产物**：`state["md_path"]` 指向转换出的 md（MinerU 的图片会解压在 md 同级的 `images/` 里 —— 这是下游 node_md_img 的硬约定）。
**外部依赖**：`common/config/mineru_config`（base_url + api_key），配置缺失直接炸。

### ③ node_md_img —— MD 图片处理（6 个 step）

**硬约定**：图片必须在 `md_path 同级 images/` 目录下（PDF 转换产物天然满足；**用户直传的 MD 若没带 images 目录，视为无图片**）。

```
step_1 校验       读 md_path → md_content（回写 state），定位 images/ 目录
      ↓           images 不存在或为空 → 打日志直接 return（⚠️ 提前返回没调 add_done_task）
step_2 扫描       遍历 images/ 里的图片文件（白名单后缀 .jpg/.jpeg/.png/.gif/.bmp/.webp），
                  正则 `![...](...图片名...)` 在 md_content 中找引用位置，
                  截取前后各 100 字符上下文 → [(图片名, Path, (上文, 下文))]
                  未被 md 引用的图片直接跳过
step_3 摘要       对每张图：base64 编码 + 上下文 → 组装 image_summary prompt
                  → 多模态模型(vl_model) 生成中文描述 → {图片名: 摘要}
                  （每次调用前 apply_api_rate_limit() 限流）
step_4 上传MinIO   先按前缀删除该文档旧图片（幂等），再逐张 fput_object，
                  拼 URL → {图片名: http地址}
step_5 替换       把 md 里的 `![原](本地路径)` 替换为 `![模型摘要](minio_url)`
                  （用 lambda 替换函数，避免摘要里的反斜杠被 re 当转义）
step_6 备份       新内容写 `{stem}_new.md` —— md_path 被改写指向 _new.md！
```

**产物**：`state["md_content"]`（图片已替换的新文本）、`state["md_path"]`（**指向 `_new.md`**）。
⚠️ 注意连带效应：下游所有"md 同名 json 兜底"的锚点从此都基于 `_new.md` 的名字。

### ④ node_document_split —— 文档切分

标题栈语义切块 → 超长剁碎 + 碎片合并 → 补 `parent_title/part` 元数据 → 落盘 `{stem}.json`。
**逐行粉碎级解析见同目录另一份文档：《node_document_split-流程详解.md》**（含 10 条隐患清单）。

**产物**：`state["chunks"]` + 磁盘备份 json。
⚠️ 注意：此时 `md_path` 是 `xxx_new.md`，所以 json 备份名为 `xxx_new.json`。

### ⑤ node_item_name_recognition —— 主体识别 + 商品名索引表

```
step_1 校验       chunks 缺失 → 用 md_path 同名 json 兜底回读；file_title 兜底
step_2 LLM识别    取前 5 个 chunk（ITEM_NAME_CONTEXT_CHUNK_K），拼 parent_title+content
                  截断到 2000 字符 → item_name_recognition prompt → LLM
                  → item_name（识别失败兜底用 file_title）
step_3 回填       每个 chunk["item_name"] = item_name（就地写入）
step_4 建表       确保 item_name_collection 存在（存在即跳过）：
                  schema: pk(auto)/file_title/item_name/dense_vector(1024)/sparse_vector
                  索引: dense→HNSW+COSINE(M=64,ef=100)  sparse→SPARSE_INVERTED_INDEX+IP
step_5 写索引表   先 delete file_title=='...' （幂等）→ 对 item_name 单条做向量化 → insert
```

**产物**：`state["item_name"]`、chunks 增加 item_name 字段、Milvus `item_name_collection` 里一条 (file_title, item_name, 向量) 记录。
**这张表的用途**：商品名 → 向量的独立索引，供检索侧做商品名召回/确认。

### ⑥ node_bge_embedding —— 批量向量化

```
step_1 校验       chunks 缺失 → json 兜底回读（回读时顺手补 item_name）
step_2 批量向量化  每 5 个 chunk 一批：
                    输入文本 = item_name + '-' + content（商品名前缀增强语义）
                    generate_embeddings(...) → dense_list + sparse_list（BGE-M3 双向量）
                    回填 chunk["dense_vector"] / chunk["sparse_vector"]
```

**产物**：`state["embeddings_content"] = chunks`（此时每个 chunk 已经是"元数据+双向量"的完整入库记录）。
代码里留了两版注释掉的实现（整批一次 / 逐条），现行版是**分批（batch=5）**折中。

### ⑦ node_import_milvus —— 写入向量库

```
step_1 校验       取 embeddings_content（⚠️ 空值校验是空壳：`if not ...: pass`）
step_2 建表       确保 chunks_collection 存在：
                  schema: chunk_id(auto)/file_title/parent_title/title/part(INT8)/
                          item_name/content(65535)/dense_vector(1024)/sparse_vector
                  索引同 item_name_collection（HNSW+COSINE / SPARSE+IP）
step_3 幂等写入    先 delete file_title=='...'（清旧） → insert(embeddings_content 整批)
```

**至此入库完成**。检索侧（query_processor）即可从 `chunks_collection` 按语义召回、从 `item_name_collection` 按商品名召回。

---

## 6. 双 Collection 设计

| | `item_name_collection`（⑤建） | `chunks_collection`（⑦建） |
|---|---|---|
| 粒度 | 一篇文档一条（商品名） | 一个 chunk 一条 |
| 主键 | pk (auto_id) | chunk_id (auto_id) |
| 业务字段 | file_title, item_name | file_title, parent_title, title, part, item_name, content |
| 向量 | dense(1024)+sparse | dense(1024)+sparse |
| 索引 | HNSW/COSINE + SPARSE/IP | HNSW/COSINE + SPARSE/IP |
| 幂等 key | file_title | file_title |
| 服务于 | 商品名召回（query侧商品名确认） | 语义检索（RAG 主数据源） |

**幂等性设计**：同 file_title 重复导入 → 先 delete 再 insert，天然覆盖旧数据，保证不重复。

---

## 7. 全链路数据流转（一个 PDF 的完整旅程）

```
上传 hak180产品安全手册.pdf
  │  API 层: task_id/local_file_path/local_dir/file_title → create_default_state
  ↓
node_entry          is_pdf_read_enabled=True, pdf_path=...
  ↓
node_pdf_to_md      MinerU 解析 → output/hak180产品安全手册/hak180产品安全手册.md (+images/)
  ↓
node_md_img         图片→MinIO→替换链接 → hak180产品安全手册_new.md
  │                 md_path 改指 _new.md
  ↓
node_document_split 标题切块+精切+合并 → chunks → hak180产品安全手册_new.json 落盘
  ↓
node_item_name_recognition   LLM → "hak180烫金机" → 回填每个 chunk → 写 item_name_collection
  ↓
node_bge_embedding  批量5 → 每个 chunk 挂上 dense/sparse 双向量
  ↓
node_import_milvus  delete 旧 + insert 全部 chunk → chunks_collection
  ↓
DONE（API 层把任务状态置 COMPLETED，可查询节点进度明细）
```

**外部依赖清单**：MinerU 云 API（PDF解析）、多模态 LLM（图片摘要）、文本 LLM（商品名识别）、BGE-M3（embedding，dense+sparse）、MinIO（图片存储）、Milvus（向量库）。每个都有独立 config：`mineru_config / lm_config / minio_config / milvus_config`。

---

## 8. ⚠️ 全流水线隐患清单（粉碎性体检）

### 8.1 结构性问题（优先级高）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | node_import_milvus.step_1 | 空值校验是空壳（`if not embeddings_content: pass`） | **空数据会继续走**：先 delete 清掉库里旧数据、再 insert 空数据 → **重复导入一个坏文件会清空该 file_title 的全部知识库数据**（数据丢失风险） |
| 2 | node_md_img 无图片分支 | 提前 `return state` **没调 `add_done_task`** | 任务进度追踪里该节点永远停在 running |
| 3 | 全线 Milvus delete | `filter=f"file_title=='{file_title}'"` 字符串拼接 | file_title 含单引号 → 语法错误/注入；建议参数化表达式 |
| 4 | node_bge_embedding | `chunks = step_1_validate_and_get_data(state)` 拿了返回值却不用，step_2 从 state 重取 | 靠 json 兜底分支里"顺手回写 state"才跑得通，逻辑耦合隐蔽；一旦重构 step_1 就断 |
| 5 | node_import_milvus | step_2/step_3 函数名都是 `prepared_item_name_collection / insert_item_name_data`（从识别节点复制粘贴），实际操作的是 chunks_collection | 名不副实，误导维护者 |
| 6 | 建表代码重复 | node_item_name_recognition.step_4 与 node_import_milvus.step_2 两份近似 schema/索引代码 | 改索引参数要改两处，漂移风险 |

### 8.2 行为性问题（优先级中）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 7 | node_entry | 不支持的扩展名 / 缺 local_file_path → 静默放行，条件边判 END | 图"正常跑完"但什么都没做，只在日志里有 warn；调用方拿到 COMPLETED 状态是假象 |
| 8 | node_md_img 的 `_new.md` 改名 | md_path 被silent改写，json 备份名、file_title 兜底值都跟着变（`xxx_new`） | 下游所有锚点带 `_new` 后缀；file_title 若靠兜底，商品名索引表的 key 也是 `xxx_new` |
| 9 | node_md_img 图片匹配 | 正则按"文件名"在 md 里找引用，若 md 引用写的是相对路径 `images/xxx.jpg`，`.*?` 能兜住；但同名不同目录图片会互相误伤 | 边缘场景替换错图 |
| 10 | node_pdf_to_md 轮询 | 600s 超时硬编码、间隔 3s 硬编码；MinerU 排队高峰期大 PDF 可能不够 | 超时即整任务失败，无重试 |
| 11 | node_item_name_recognition | 只看前 5 个 chunk / 2000 字符识别商品名 | 长文档开头是目录/封面时识别质量差 |
| 12 | main_graph `__main__` | `final_state = None` 后 stream 循环被注释，invoke 返回值没接 → 打印的永远是 None | 本地测试输出误导 |
| 13 | node_md_img_practice.py | 练习文件混在 nodes/ 里，还 import 了 lib2to3/multipart/networkx 等无关库 | 应移出或删除，避免误导 |

### 8.3 一致性小问题

- node_pdf_to_md 错误文案有错别字（"pdf_path的呵值为空,无法读取呵呵文件"）；
- node_pdf_to_md 里 `local_dir` 兜底赋的是 `Path` 对象而正常路径传的是 str，类型不一致（state 里两种类型并存）；
- node_md_img / node_item_name_recognition 末尾都有不可达的重复 `return state`（死代码）。

---

## 9. 调用方

| 调用方 | 方式 |
|---|---|
| `api/file_import_service.py`（生产入口） | FastAPI 后台任务：`run_import_task(state)` 里 `kb_import_app.stream(state, stream_mode="updates")` 逐节点推进，节点完成即日志打点；成功/失败回写任务状态（PROCESSING→COMPLETED/FAILED） |
| `test/fastapi_bk/test_fast_api.py` | 同款 stream 用法的测试 |
| `main_graph.py __main__` | 本地直跑 `invoke`（输出有 bug，见 #12） |
| 各 node 的 `__main__` | 每个节点都可单独跑（构造最小 state），联调用 |

---

## 10. 一句话总结

> **一条以 `ImportGraphState` 为总线、`node_entry` 按扩展名二分（PDF 走 MinerU 转换 / MD 直传）后在 `node_md_img` 汇合的 LangGraph 直线流水线：图片进 MinIO 换摘要链接 → 标题栈切 chunks → LLM 识别商品名并建索引表 → BGE-M3 双向量 → 按 file_title 幂等写入 Milvus 双 collection —— 标准范式是"node_log 包节点、step_log 包步骤、task_utils 记进度、json 兜底可断点重入"。**

最需要优先修的三件事：**① node_import_milvus 的空数据会清库（#1）；② node_md_img 提前返回漏 done_task（#2）；③ Milvus filter 字符串拼接（#3）**。
