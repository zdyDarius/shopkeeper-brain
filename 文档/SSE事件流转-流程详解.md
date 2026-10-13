# SSE 事件流转 粉碎性流程解析

> 核心文件：`utils/sse_utils.py`（四件套：`SSEEvent` / `create_sse_queue` / `push_to_session` / `sse_generator`）
> 使用方：`api/query_service.py`（查询服务的流式应答）、`utils/task_utils.py`（节点进度推送）、`processor/query_processor/nodes/node_answer_output.py`（答案逐字输出）、`api/dist`（前端 EventSource 消费）
> 一句话定位：**用"每个 session 一条 `queue.Queue`"做跨线程邮箱，后台图执行线程往邮箱塞事件，FastAPI 事件循环里的 SSE 生成器从邮箱取事件，按 SSE 协议写给浏览器**。
> 更新日期：2026-10-12

---

## 1. 一图流：全链路时序

```
 前端(EventSource)          FastAPI(事件循环)              后台线程(线程池)                queue(邮箱)
 │                          │                             │                             │
 │ ①POST /query ───────────→│                             │                             │
 │   {query,session_id,     ├─ create_sse_queue(sid) ───────────────────────────────→ 注册新队列
 │     is_stream:true}      ├─ BackgroundTasks.add(get_query_graph)                   │
 │                          │   (响应返回后才触发)          │                             │
 │ ←─ {session_id} ─────────┤                             │                             │
 │                          │                             │                             │
 │ ②GET /stream/{sid} ─────→│                             │                             │
 │                          ├─ StreamingResponse(sse_generator(sid,request))          │
 │ ←─ event: ready ─────────┤   (生成器启动即发)            │                             │
 │                          │                             │                             │
 │                          │                             ├ clear_task / 状态置processing │
 │                          │                             ├→ push progress(processing) ─→│
 │                          │                             │                             │
 │                          │                             ├ 图节点开始 add_running_task  │
 │                          │                             ├→ push progress ───────────→│
 │                          │ ←─ event: progress ─────────│  (生成器逐条取出打包)         │
 │                          │                             ├ 图节点结束 add_done_task     │
 │                          │ ←─ event: progress ─────────├→ push progress ───────────→│
 │                          │                             │  ... (每个节点 2 条) ...     │
 │                          │                             │                             │
 │                          │                             ├ node_answer_output:         │
 │                          │ ←─ event: delta ────────────├→ push delta {"delta":"字"} ─│
 │                          │ ←─ event: delta ────────────├→ ... 逐字推送 sleep(0.03)   │
 │                          │                             │                             │
 │                          │                             ├ 图结束 状态置completed        │
 │                          │ ←─ event: progress ─────────├→ push progress(completed) ─→│
 │                          │                             ├ push final(完整答案+图片) ──→│
 │ ←─ event: final ─────────┤                             │                             │
 │                          │                             │                             │
 │ ③前端收到final后关闭连接    │                             │                             │
 │   (EventSource.close())  │                             │                             │
 │ ── 断开 ─────────────────→│ finally: remove_sse_queue(sid)                          │
```

> ⚠️ 关键事实：**没有任何代码推送 `__close__`**，流的结束完全依赖前端收到 `final` 后主动断开（见 §8）。

---

## 2. sse_utils.py 四件套逐一粉碎

### 2.1 `SSEEvent` —— 事件类型常量表

```python
class SSEEvent:
    READY    = "ready"       # 连接建立确认
    PROGRESS = "progress"    # 任务/节点进度
    DELTA    = "delta"       # LLM 流式输出增量
    FINAL    = "final"       # 最终完整答案
    ERROR    = "error"       # 错误信息
    CLOSE    = "__close__"   # 关闭连接信号（⚠️ 定义了但无人发送）
```

### 2.2 `_session_stream` —— 全局会话注册表

```python
_session_stream: Dict[str, queue.Queue] = {}   # 模块级单例字典
```

- **Key = session_id，Value = 该会话专属的 `queue.Queue`**（无界、线程安全）。
- 这是整个 SSE 机制的**枢纽数据结构**：生产者（后台线程）和消费者（事件循环）唯一的交汇点。
- 生命周期三函数：

| 函数 | 做什么 | 谁调用 |
|---|---|---|
| `create_sse_queue(sid)` | `queue.Queue()` 注册进字典并返回 | `POST /query`（is_stream 时） |
| `get_sse_queue(sid)` | 查字典，无则 None | push_to_session / sse_generator |
| `remove_sse_queue(sid)` | `pop(sid, None)` 摘除 | **只有 sse_generator 的 finally** |

⚠️ `create_sse_queue` 对已有 session 是**覆盖式重建**（新 Queue 顶掉旧 Queue）——这既是"同会话重复提问时清掉旧事件"的机制，也埋了"幽灵生成器"隐患（§9-3）。

### 2.3 `push_to_session` —— 生产者统一入口

```python
def push_to_session(session_id, event, data: dict):
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})   # 队列里存的是 dict，不是 SSE 文本
    else:
        print(f"[SSE] Warning: No queue found ...")         # 队列不存在 → 只打警告，事件丢弃
```

三个要点：
1. **队列里存的是 `{"event": str, "data": dict}`**，JSON 序列化推迟到消费端（`_sse_pack`）——生产端零格式负担；
2. **队列不存在时静默丢弃**（只 print 警告）——不抛异常，生产者完全无感；
3. 线程安全由 `queue.Queue` 保证 → 任何线程都能直接调。

### 2.4 `sse_generator` —— 消费者（SSE 生成器）

逐段粉碎：

```python
async def sse_generator(session_id, request):
```

**(a) 启动检查**：`get_sse_queue(session_id)` 为 None（前端连了 `/stream` 但没先 POST /query）→ 直接 return，**流内容为空但 HTTP 200**，日志列出当前所有可用 session。

**(b) ready 事件**：进循环前**无条件先 yield 一条 `ready`**——前端确认"管道通了"的信号（当前前端并未监听它）。

**(c) 主循环**（每轮四步）：

```python
while True:
    # ① 断开检测：每轮最多 1s 粒度感知客户端断开
    if await request.is_disconnected(): break

    # ② 跨线程桥接：把阻塞的 queue.get 塞进默认线程池执行，不卡事件循环
    msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
    # ↓ 等价于 stream_queue.get(block=True, timeout=1.0)
    # ↓ 队列空 → queue.Empty → continue 下一轮（回到断开检测）

    # ③ 关闭信号：收到 __close__ 就退出（当前无人发送，见 §8）
    if event == "__close__": break

    # ④ 打包并写出
    yield _sse_pack(event, data)
```

**(d) 异常矩阵**：

| 异常 | 处理 |
|---|---|
| `asyncio.CancelledError`（客户端断开时 Starlette 取消生成器） | 捕获、静默 return |
| `ConnectionResetError` / `BrokenPipeError`（写给已断的 socket） | 同上 |
| 其他 `Exception` | print 后走出 try（依然会进 finally） |

**(e) finally**：`remove_sse_queue(session_id)` —— **整个生命周期里唯一的队列摘除点**。无论正常结束、断开、异常，队列一定会被摘掉。

### 2.5 `_sse_pack` —— SSE 线格式

```python
f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

产出的字节流形如：

```
event: progress
data: {"status": "processing", "done_list": ["确认问题产品"], "running_list": ["切片搜索"]}

```

- **命名事件**（`event:` 行）让前端可以按类型 `addEventListener("delta", ...)`，而不是挤在 `onmessage` 里 switch；
- **结尾必须空行 `\n\n`**——SSE 协议的事件分隔符，少了它浏览器会把两条事件黏成一条不派发；
- `ensure_ascii=False` —— 中文答案原样传输。

---

## 3. 线程模型：一条队列如何跨两个世界

```
┌─────────────────────────────────┐      ┌──────────────────────────────────────┐
│  asyncio 事件循环 (uvicorn)      │      │  AnyIO 线程池                          │
│                                 │      │                                      │
│  GET /stream → sse_generator    │      │  POST /query 响应返回后：               │
│    ↑  await run_in_executor ──────────→ │    get_query_graph()                  │
│    │  (queue.get 借线程池执行)    │      │      └ query_app.invoke(state)        │
│    │                            │      │          └ 各节点函数（同步代码）        │
│  yield SSE 文本 ──→ 浏览器       │      │              ├ push_to_session ───┐  │
│                                 │      │              └ add_*_task ────────┤  │
└─────────────────────────────────┘      └──────────────────────────────────┼──┘
                                                          ┌─────────────────▼───┐
                       queue.Queue（线程安全邮箱，生产者多线程、消费者单协程）│
                                                          └─────────────────────┘
```

- **生产端**：`BackgroundTasks` 里的同步函数 `get_query_graph` 在**线程池线程**里跑 LangGraph，节点里所有 `push_to_session` 都发生在线程池；
- **消费端**：`sse_generator` 是**事件循环里的协程**，绝不能直接 `queue.get()` 阻塞（会卡死整个 API）——所以用 `run_in_executor` 把阻塞 get 借给线程池，每次最多借 1 秒；
- `queue.Queue` 本身线程安全 → 两侧无锁通信。

⚠️ 代价：**每条活跃 SSE 流会周期性占用默认线程池一个 worker**（每次最多 1s）。并发流多了会挤占 `run_in_executor` 的其他使用者（默认线程池大小 ≈ min(32, CPU+4)）。

---

## 4. 事件类型全表（谁发、何时发、装什么）

| event | 生产者 | 触发时机 | payload | 前端是否监听 |
|---|---|---|---|---|
| `ready` | sse_generator | 生成器启动，进主循环前 | `{}` | ❌（未监听） |
| `progress` | `task_utils.task_push_queue` | 图节点开始/结束；任务状态变化 | `{status, done_list[], running_list[]}`（节点中文名） | ✅ |
| `delta` | `node_answer_output` | 答案逐字推送（每字 `sleep(0.03)`） | `{"delta": "单字"}` | ✅ |
| `final` | `query_service.get_query_graph` | 图成功跑完后 | `{answer, status:"completed", image_urls[], name:"zdy"}` | ✅ |
| `error` | `query_service.get_query_graph` | 图执行抛异常 | `{error: "异常信息"}` | ✅ |
| `__close__` | **（无）** | **——** | —— | ——（协议内部信号） |

> 前端 bundle（`api/dist/assets/ChatView-*.js`）实际监听：`delta / progress / final / error / final_answer`。其中 **`final_answer` 后端从不发送——死监听**。

---

## 5. 三条推送链路粉碎

### 链路 A：`progress`（进度链）—— 节点 → task_utils → 队列

```
图节点开头: add_running_task(session_id, 节点名, is_stream)
图节点结尾: add_done_task(session_id, 节点名, is_stream)
                │ is_stream=True 时
                ↓
        task_push_queue(task_id=session_id)
                ↓
        push_to_session(sid, "progress", {
            "status": get_task_status(sid),        # processing/completed/failed
            "done_list":  [...中文名...],           # _NODE_NAME_TO_CN 映射
            "running_list": [...中文名...]
        })
```

关键细节：
1. **query 图的 7 个节点全部传了 `is_stream`**（`state["is_stream"]` 或 `state.get("is_stream")`）→ 每节点产生 2 条 progress（开始 1 条 + 结束 1 条）；**import 流水线节点没传**（默认 False）→ 导入流程不发 SSE。
2. 节点名 → 中文：`_NODE_NAME_TO_CN` 映射表（`node_item_name_confirm → "确认问题产品"` 等）；**key 必须与 add_node 的节点名一致**，新增节点忘了登记就发英文原名。
3. `add_done_task` 会先把同名节点从 running 列表移除、再追加进 done 列表 → 前端能画出"节点逐个点亮"的进度条。
4. **位置参数巧合**：`get_query_graph` 里 `update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)` —— 第三个形参其实叫 `push_queue`，is_stream 恰好传了进去 → 流式模式下状态变化也会即时推 progress（processing / completed 各一条）。语义碰巧对，纯靠参数位置。

### 链路 B：`delta`（打字机链）—— node_answer_output 逐字推

```python
for ch in base_answer:
    push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
    time.sleep(0.03)
```

关键细节：
1. **假流式**：`base_answer = state.get("answer") or 演示用测试回答` —— 文本在推送前就已经完整生成，逐字推送 + 0.03s/字 只是打字机特效（约 33 字/秒，500 字答案 ≈ 15 秒）。真实 LLM 流式应接 token 级回调，当前是演示形态。
2. **DELTA 与 FINAL 内容重叠**：逐字推完 delta 后，API 层又会推一条 final 带完整 answer —— 前端通常"delta 累积渲染、final 收口"，需要自己处理二者关系。
3. sleep 发生在**后台线程**里（不卡事件循环），但意味着图在这一步被人为拖长。

### 链路 C：`final` / `error`（收口链）—— API 层统一收口

```
get_query_graph（后台线程）:
  try:
    clear_task(sid) → update_task_status(PROCESSING, push=True)
    query_app.invoke(state)
    update_task_status(COMPLETED, push=True)   → 先推 progress(completed)
    push FINAL {answer, image_urls, name:'zdy'} → 再推 final
  except:
    update_task_status(FAILED)                 → status 只写不推
    push ERROR {error: str(e)}                 → 推 error
```

关键细节：
1. **收尾顺序是 progress(completed) → final**，前端两个监听先后都会触发；
2. FINAL 里的 `'name': 'zdy'` 是**硬编码测试字段**混进了生产 payload；
3. ERROR 推送后 `get_query_graph` **吞掉异常返回 None**：
   - 流式路径：无所谓（结果走 SSE）；
   - 非流式路径：`final_state.get("answer")` 直接 `AttributeError` → 500，而不是优雅的错误 JSON（§9-11）。

---

## 6. 一次真实流式查询的完整事件序列

以 `POST /query {query:"hak180怎么换刀", session_id:"s1", is_stream:true}` 为例，前端按序收到：

```
event: ready      data: {}
event: progress   data: {"status":"processing","done_list":[],"running_list":["确认问题产品"]}
event: progress   data: {"status":"processing","done_list":["确认问题产品"],"running_list":[]}
event: progress   data: {"status":"processing","done_list":["确认问题产品"],"running_list":["切片搜索"]}
event: progress   data: {...切片搜索完成, running_list:[]}            (embedding/hyde/web_search 三路)
event: progress   data: {...,"running_list":["倒排融合"]}
event: progress   data: {...,"running_list":["重排序"]}
event: progress   data: {...,"running_list":["生成答案"]}
event: delta      data: {"delta":"h"}
event: delta      data: {"delta":"a"}
event: delta      data: {"delta":"k"}
...（逐字,~33字/秒）
event: progress   data: {"status":"completed","done_list":[...全部6个节点...],"running_list":[]}
event: final      data: {"answer":"...完整答案...","status":"completed","image_urls":[...],"name":"zdy"}
（之后流保持打开,直到前端 close() —— 服务端不会主动结束）
```

（多路检索路径下 progress 条数 = 每节点 2 条 × 6 节点 + 状态变化 2 条；若 `node_item_name_confirm` 直接产出 answer，则走直答路径，只有 2 节点。）

---

## 7. 前端消费契约（api/dist）

- `GET /chat.html` 返回打包好的 Vue 前端（`dist/index.html`）；
- 前端流程：`POST /query` 拿 session_id → `new EventSource("/stream/{session_id}")` → 分类型监听；
- 实测 bundle 监听：`progress`（进度条）、`delta`（打字机追加）、`final`（收口渲染 + 图片）、`error`（报错提示）、`final_answer`（**死监听**，后端从不发此事件名）；
- `ready` 未被监听——服务端发了但前端不消费（调试期遗留）。

---

## 8. 流的四条终止路径

| # | 终止路径 | 触发条件 | 后果 |
|---|---|---|---|
| 1 | **`__close__` 事件** | 需要有人 `push_to_session(sid, SSEEvent.CLOSE, ...)` —— **当前代码没有任何地方推送** | 设计上的优雅关闭通道，实际闲置 |
| 2 | **客户端断开检测** | 前端 `EventSource.close()` / 关页 → 下一轮 `request.is_disconnected()` 为真（**最多延迟 1 秒**，因为要等 queue.get 超时回来） | 当前**唯一的实际关闭方式**：前端收到 final 后自行断开 |
| 3 | **写失败/取消** | `CancelledError / ConnectionResetError / BrokenPipeError` | 捕获后静默退出 |
| 4 | **异常兜底** | 其他 Exception | print 后退出 |

四条路径最终都汇入 **finally → `remove_sse_queue(sid)`**，队列必被摘除。

⚠️ 推论：
- **final 推完后流不会关**：如果前端收到 final 后不断开，连接会一直挂着（每秒轮询一次队列+断开检测），直到客户端超时或主动断开；
- **`SSEEvent.CLOSE` 是死代码**：合理的补法是 `get_query_graph` 在推完 final/error 后 `push_to_session(sid, SSEEvent.CLOSE, {})`，让服务端主动收流。

---

## 9. ⚠️ 隐患清单（粉碎性体检）

| # | 位置 | 问题 | 影响 |
|---|---|---|---|
| 1 | 全链路 | **无人推送 `__close__`** | final 之后流不主动关闭，依赖前端自觉断开；前端忘记 close → 连接悬挂 + 队列常驻 |
| 2 | 生命周期 | **队列唯一摘除点在 generator 的 finally**。客户端 POST /query 后**从不连 /stream** → 队列永远留在 `_session_stream`，后台照常往里塞满全部事件 | 内存泄漏（`queue.Queue` 无界）；同一进程长期运行会累积幽灵会话 |
| 3 | create_sse_queue | **覆盖式重建**：同 session_id 重复 POST，新 Queue 顶掉旧的；若旧流的 generator 还活着，它持有的是**旧 Queue 引用**，继续 drain 空队列 | "幽灵生成器"：旧连接永远收不到新事件（新事件全进新队列），表现为前端卡死 |
| 4 | 断开语义 | 客户端断开 → finally 摘队列 → **后台图不知道，继续跑完全部节点**；此后每次 `push_to_session` 只打警告 | 算力浪费；无取消传播（LangGraph 不支持外部取消） |
| 5 | sse_generator | **无心跳**：两事件之间长时间空转时不发任何字节（连 SSE 注释行 `: ping` 都没有） | 中间代理/网关（nginx 等）可能掐断空闲连接，长查询（逐字 sleep 拖 15s+）期间有风险——目前 delta 频繁尚可，纯检索阶段有空窗 |
| 6 | run_in_executor | 每条活跃流周期性占用默认线程池 worker（每次 ≤1s） | 并发 SSE 一多，挤占全局线程池；规模化应换 `asyncio.Queue` 或 aiohttp 背压方案 |
| 7 | get_query_graph | `update_task_status(sid, status, is_stream)` 第三参是 `push_queue`，is_stream 纯靠**位置参数巧合**对上语义 | 一旦有人改签名/加参数立刻错位，建议显式 `push_queue=is_stream` |
| 8 | FINAL payload | `'name': 'zdy'` 硬编码 | 测试字段污染生产协议 |
| 9 | node_answer_output | 假流式：答案先整体生成再逐字 sleep(0.03) | 演示可用；接真 LLM 时应改为 token 级回调直推 DELTA |
| 10 | 前端 | 监听 `final_answer` 但后端从不发 | 死代码/曾改名的遗留，易误导联调 |
| 11 | /query 非流式 | 失败路径 get_query_graph 吞异常返回 None → `final_state.get("answer")` AttributeError | 非流式查询失败时用户看到 500 堆栈而非错误 JSON |
| 12 | sse_utils | 全部用 `print` 不走项目 logger | 生产日志无结构、无 trace_id，难排查 |
| 13 | 队列启动竞态 | 后台任务可能在 GET /stream 之前就开始推事件 | **非 bug**：queue 在 POST 时已创建，早期事件先进队列缓冲，generator 启动后统一倒出——设计自洽（写进文档防止误判） |

---

## 10. 与导入流水线的关系（一图两用）

同一套 sse_utils 也可服务 import 流水线：`task_utils` 是公共模块，import 的节点调用 `add_running_task/add_done_task` 时**没传 is_stream**（默认 False）→ 当前导入流程不发 SSE。若想让上传导入也走进度推送，只需在 `api/file_import_service.py` 建 queue + import 节点把 `is_stream` 传进 `add_*_task` 即可——机制层面零改动。

---

## 11. 一句话总结

> **`POST /query` 先建 `queue.Queue` 邮箱再派后台线程跑图；图节点经 `task_utils` 发 progress、答案节点逐字发 delta、API 层收口发 final/error，全部投进邮箱；`GET /stream` 的 SSE 生成器用 `run_in_executor` 每秒从邮箱捞一次，按 `event:\ndata:\n\n` 写给浏览器，直到前端断开才在 finally 里摘掉邮箱 —— 一个"queue 桥接线程池与事件循环"的最小可用 SSE 方案，短板在无人发 `__close__`、队列无兜底回收、以及前端断开后后台照跑。**

最值得先修的三件事：**① final/error 后补推 `SSEEvent.CLOSE`（#1）；② 给 `_session_stream` 加会话兜底回收/TTL（#2）；③ 显式 `push_queue=is_stream`（#7）。**
