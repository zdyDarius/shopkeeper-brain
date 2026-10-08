# node_document_split.py 粉碎性流程解析

> 文件位置：`processor/import_processor/nodes/node_document_split.py`
> 节点职责：**把一份完整的 Markdown 文档切成一批大小合适、语义完整、带元数据的 Chunk**，供后续向量化 / 检索使用。
> 更新日期：2026-10-08

---

## 1. 在整体 Pipeline 中的位置

```
node_entry → node_pdf_to_md / node_md_img → 【node_document_split】 → (向量化/入库 节点)
```

- 上游产物：`md_path`（md 文件路径）、`md_content`（可选，md 文本）、`file_title`（可选，文件标题）
- 本节点产物：`state["chunks"]`（Chunk 字典列表）+ 磁盘备份 `<md同名>.json`
- 下游消费：`chunks` 将被嵌入模型向量化（对应 state 中的 `embeddings_content` 字段）

---

## 2. 全局常量（切分策略的"尺码表"）

| 常量 | 值 | 作用 |
|---|---|---|
| `CHUNK_MAX_SIZE` | 1000 | 单块**硬上限**，超过会向量失真 |
| `CHUNK_SIZE` | 600 | 单块**理想大小**，超过触发二次切分 |
| `CHUNK_OVERLAP` | 50 | 相邻块重叠长度（⚠️ 注释写的是 20，与实际值不符；且当前实际未生效，见 §9） |
| `CHUNK_MIN` | 400 | **碎片阈值**，低于它判定为短碎片、尝试与邻居合并 |

设计意图：`400 ~ 600 ~ 1000` 三道闸 —— 太长的剁开（>600 切），太短的拼起来（<400 合），拼也不能拼爆（<1000）。

---

## 3. 主流程总览（node_document_split）

```
┌─────────────────────────────────────────────────────────┐
│  add_running_task(task_id, "node_document_split")       │
│                    ↓                                    │
│  step_1  validate_get_data      取数 + 校验 + 换行归一   │
│                    ↓                                    │
│  step_2  split_document_by_title 按标题切（保语义）       │
│                    ↓                                    │
│  step_3  refine_split_and_merge  超长剁碎 + 碎片合并      │
│                    ↓                                    │
│  step_4  padding_chunks_metadata 补 part/parent_title   │
│                    ↓                                    │
│  step_5  backup_chunks_json     落盘备份 JSON            │
│                    ↓                                    │
│  state["chunks"] = refine_chunks                        │
│  add_done_task(task_id, "node_document_split")          │
└─────────────────────────────────────────────────────────┘
```

入口处 `@node_log` 装饰器自动记录节点开始/完成/异常 + 耗时；每个 step 函数上的 `@step_log` 同理（步骤级），异常不吞、继续上抛。

---

## 4. step_1 —— 数据校验与获取

**签名**：`step_1_validate_get_data(state) -> (md_content, file_title, md_path)`

做了 4 件事：

1. **取三件套**：`md_content` / `md_path` / `file_title`。
2. **内容兜底**：`md_content` 为空时尝试从 `md_path` 读文件（utf-8）并回写 state；
   两者都没有 → `logger.error` + **抛 `ValueError` 直接终止业务**（这是全节点唯一的硬失败点）。
3. **标题兜底**：`file_title` 为空 → 取 md 文件名（去扩展名），再空则 `"default"`，回写 state 并 warn。
4. **换行符归一**：`\r\n` / `\r` → `\n`，因为后续全程**按行处理**，必须抹平 OS 差异。

> 注意：归一化后的 `md_content` **没有回写 state**（只有从文件读的那条路径回写了原始内容）。下游用的是返回值，所以无实际影响，但 state 里的内容和实际处理的内容可能不一致。

---

## 5. step_2 —— 按标题切分（核心状态机，保语义）

**签名**：`step_2_split_document_by_title(md_content, file_title) -> list[chunk]`

产出 chunk 结构：`{"title": 标题路径, "file_title": 文件名, "content": 标题+正文}`

### 5.1 状态机的 5 个状态变量

| 变量 | 含义 |
|---|---|
| `current_title` | 当前"激活"的标题路径（如 `# 安全_## 电气安全`），空串 = 还没遇到任何标题 |
| `current_content_lines` | 当前标题下的正文缓存 |
| `orphan_lines` | **孤儿行缓存**：出现在第一个标题之前的内容 |
| `is_code` | 代码块标记，遇到 ```` ``` ```` 或 `~~~` 翻转 |
| `heading_stack` | 标题栈，维护 1~6 级标题的层级路径 |

标题识别正则：`^#{1,6}\s.+`（`#` 与文字之间**必须有空格**）。

### 5.2 逐行处理逻辑

```
每行 → strip 后为空？ → 丢弃（⚠️ 段落间的空行信息全部丢失）
     → 是代码围栏(```/~~~)？ → 翻转 is_code，围栏行本身也入缓存
     → is_code == True？     → 原样入缓存（不参与标题判断，保护代码里的 # 注释）
     → 匹配标题正则？        → 【结算旧块】+【维护标题栈】+【激活新标题】
     → 否则                  → 普通正文行，入缓存（无激活标题则进孤儿区）
```

### 5.3 标题结算（遇到新标题时）

- 条件：`current_title` 非空 **且** `current_content_lines` 非空 → 才把上一块封盘。
  - ⚠️ 推论：**连续两个标题（空章节）会被静默丢弃**，不产生 chunk。
- 封盘前若孤儿区有内容 → **把孤儿行 prepend 到当前块正文前面**（孤儿内容被"过继"给第一个有正文的标题块）。
- chunk 的 content 格式：`标题路径\n正文行1\n正文行2...`

### 5.4 标题栈维护（生成层级标题路径）

```python
heading_level = # 号个数
while len(heading_stack) < heading_level: heading_stack.append(None)  # 跳级补 None
heading_stack = heading_stack[:heading_level]                          # 砍掉更深层级
heading_stack[heading_level-1] = 当前标题
current_title = '_'.join(过滤 None 后的栈)                              # 例: "# 安全_## 电气安全"
```

效果：每个 chunk 的 `title` 是**从根到当前的完整标题路径**，天然携带层级上下文。

### 5.5 循环结束后的尾部清算

- 还有激活标题且（正文或孤儿非空）→ 封最后一块；
- 全文**没有任何标题**、只有孤儿 → 以 `file_title` 作为标题单独成块（整篇一块）。

---

## 6. step_3 —— 精细化：超长剁碎 + 碎片合并（保大小）

**签名**：`step_3_refine_split_and_merge_chunks(title_chunks) -> list[chunk]`

两道工序，**先切后合**。

### 6.1 第一刀：`_split_chunk_content`（>600 的块）

对每个 `len(content) > CHUNK_SIZE` 的块：

1. `prefix = title + "\n"`；
2. `deal_content = content.lstrip(prefix)` —— ⚠️ **这里有 bug**（见 §9-2），意图是剥掉开头的标题行；
3. 创建 `RecursiveCharacterTextSplitter(chunk_size=600-len(prefix), overlap=50, separators=["\n\n","\n","。","！","？","；","，"," "])` —— ⚠️ **创建了但从未使用**；
4. 实际行为：`deal_content.splitlines()` **按行硬切，一行一个子块**：

```python
{
  "file_title": 继承,
  "parent_title": 父块title,
  "title": f"{父块title}_{index}",   # 从 1 开始编号
  "part": index,
  "content": prefix + 该行           # 每个子块都重新带上标题前缀
}
```

### 6.2 第二刀：`_merge_chunk_content`（<400 的碎片合并）

合并三前提（代码注释原文）：

1. **同一个父标题下**（`parent_title` 相等且非空）；
2. 基准块内容 `< CHUNK_MIN(400)`；
3. 合并后 `< CHUNK_MAX_SIZE(1000)`。

算法（单指针 + 基准块）：

```
base = 第一个块
for next in 剩余块:
    base 长度 > 400?  → base 直接出锅, base = next
    否则:
        base.parent_title 存在且 == next.parent_title?
            是 → 去掉 next 的标题前缀(len(parent_title)+1 个字符)后拼到 base 尾部
                 拼接后 > 1000? → 不拼, base 出锅, base = next
            否 → base 出锅, base = next
最后 base 出锅
```

特点：
- 合并时会**剥掉被并块的标题前缀**（`content[len(parent_title)+1:]`），避免标题重复堆叠；
- `base_chunk` 是**就地累加**的，可以连续吞并多个邻居（只要每次都不超 1000）；
- ⚠️ 关键限制：`is_same_parent_title` 要求 `base_chunk.get("parent_title")` **truthy**。而 `parent_title` 只有被 `_split_chunk_content` 切出来的子块才有（step_2 直出的块要到 step_4 才补），所以——
  **step_2 直出的小块之间永远不会合并，只有"同一大块切出的子块"之间才会合并**（见 §9-4）。

---

## 7. step_4 —— 元数据补齐（保可追溯）

对每个 chunk 就地补两个字段：

| 字段 | 规则 |
|---|---|
| `parent_title` | 缺失时 = 自己的 `title`（没被切过的块，父标题就是自己） |
| `part` | 缺失时 = 1（未被切分的块视为第 1 部分） |

执行后所有 chunk 字段齐备：**`file_title / title / parent_title / part / content`** 五件套。

---

## 8. step_5 —— 落盘备份

- 输出路径：与 md 文件**同目录、同主名**的 `.json`（如 `xxx.md` → `xxx.json`）；
- 格式：`json.dumps(chunks, ensure_ascii=False, indent=4)`，utf-8；
- 用途：调试回溯 + 下游可直接复用（对 `output/hak180产品安全手册/hak180产品安全手册.json` 这种产物就是它）。

---

## 9. ⚠️ 已知问题 / 隐患清单（粉碎性体检）

| # | 位置 | 问题 | 影响 | 建议 |
|---|---|---|---|---|
| 1 | `_split_chunk_content` | 创建了 `RecursiveCharacterTextSplitter` 但**从未调用**，实际是按行 `splitlines()` 硬切 | `CHUNK_SIZE/OVERLAP/separators` 配置全部失效；长行没人管（一行可能远超 600），短行产生大量碎片再靠 merge 兜底 | 真正调用 `splitter.split_text(deal_content)` |
| 2 | `_split_chunk_content` | `content.lstrip(prefix)` 误用 —— `lstrip` 参数是**字符集合**不是前缀串 | 会把开头所有"碰巧在 prefix 字符集里"的字符全吃掉，可能误删正文 | 改用 `content.removeprefix(prefix)` |
| 3 | 文件头注释 | `CHUNK_OVERLAP` 注释写"重叠 20 字符"，实际值 50 | 注释误导 | 改注释 |
| 4 | `_merge_chunk_content` | `parent_title` 判断要求 truthy，step_2 直出块在 step_4 前没有该字段 | step_2 直出的小块（<400）**永远合并不了**，与注释意图"碎片合并"不符 | 把 step_4 的补字段提前到 step_3 之前，或 merge 时回退比较 `title` |
| 5 | step_2 | 空行全部 `continue` 丢弃 | 段落边界信息丢失，正文黏连；也影响后续按 `\n\n` 切分的效果 | 视情况保留段落空行 |
| 6 | step_2 | `current_title` 非空但正文为空时**不结算** | 连续标题（空章节）静默丢失 | 如需保留章节骨架，空章节也应成块 |
| 7 | step_2 | 标题正则 `^#{1,6}\s.+` 要求 `#` 后必须有空格 | `#Title` 这种无空格写法识别不到，会当正文 | 正则改 `^#{1,6}\s*.+` 或按 CommonMark 规范决策 |
| 8 | `_merge_chunk_content` | `next_cleared_content = content[len(parent_title)+1:]` 强假设 content 以 `parent_title\n` 开头 | 若未来 content 结构变化会切错 | 用 `removeprefix` + 断言更稳 |
| 9 | step_1 | 换行归一化后的 content 未回写 state | state 里存的还是未归一内容（仅从文件读的分支回写了原始内容） | 统一在末尾回写 |
| 10 | step_2 | 代码块内空行被丢弃 + 围栏行入缓存但块内行用原始行 | 行为不完全一致，但影响极小 | 可忽略 |

---

## 10. Chunk 数据结构的一生

| 阶段 | file_title | title | content | parent_title | part |
|---|---|---|---|---|---|
| step_2 产出 | ✅ | ✅ 标题路径 | ✅ `title\n正文` | ❌ | ❌ |
| step_3 切出的子块 | ✅ | ✅ `父title_N` | ✅ `父title\n该行` | ✅ 父title | ✅ N |
| step_3 未切/合并后 | ✅ | 不变 | 可能变长 | 子块有/直出块无 | 子块有/直出块无 |
| step_4 之后（终态） | ✅ | ✅ | ✅ | ✅（直出块=自身title） | ✅（默认1） |

## 11. State 读写契约

**读**：`task_id`、`md_content`、`md_path`、`file_title`
**写**：`md_content`（仅文件读取分支）、`file_title`（仅兜底分支）、`chunks`（最终产物）

**硬失败条件**：`md_content` 为空且 `md_path` 无效 → `ValueError`，节点终止。

## 12. 本地测试入口（`__main__`）

- 依赖：`output/hak180产品安全手册/hak180产品安全手册.md` 存在；
- 构造最小 state（`md_path/task_id/file_title/local_dir`），直接跑 `node_document_split`；
- 代码里预留了先跑 `node_md_img`（图片处理节点）做端到端联调的位置（当前注释掉）。

---

## 13. 一句话总结

> **以 Markdown 标题层级为骨架做语义切分（标题栈拼路径、代码块不拆、孤儿内容过继），再对超 600 的块按行剁碎、对不足 400 的同源碎片合并（上限 1000），最后补齐 `parent_title/part` 元数据并落盘 JSON —— 一套"先保语义、再保大小、后保追溯"的三段式切分流水线。**
