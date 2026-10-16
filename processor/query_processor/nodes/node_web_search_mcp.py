import asyncio
import json
import time
import sys

import aiohttp
from agents.mcp import MCPServerStreamableHttp

from common.config.bailian_mcp_config import mcp_config
from processor.query_processor.state import QueryGraphState
from utils.task_utils import add_done_task,add_running_task
from common.logging.logger import logger, node_log, step_log


@node_log("node_web_search_mcp")
def node_web_search_mcp(state:QueryGraphState):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    # TODO 1. 添加节点到运行时列表
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # TODO 2. 从状态中获取数据并进行校验
    rewritten_query: str = step_1_validate_and_get_data(state)
    # TODO 3. 连接mcp服务器进行工具的调用   启动事件循环  执行协程的异步操作
    result = asyncio.run(step_2_call_mcp_tool(rewritten_query))
    # TODO 4. 处理返回结果(格式化)
    text_result: str = result.content[0].text
    web_search_docs = json.loads(text_result).get("pages", [])
    # TODO 5. 添加节点到已完成列表
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    return {"web_search_docs": web_search_docs}
@step_log("step_1_validate_and_get_data")
def step_1_validate_and_get_data(state:QueryGraphState):
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error(f"重写的问题为空,业务无法继续,提前终止!")
        raise ValueError(f"重写的问题为空,业务无法继续,提前终止!")
    return rewritten_query


async def step_2_call_mcp_tool(rewritten_query:str):
    mcp_server = MCPServerStreamableHttp(
        name="Streamable HTTP Python Server",
        params = {
            'url': mcp_config.mcp_base_url,
            'headers':{
                "Authorization": f"Bearer {mcp_config.api_key}"
            },
            "timeout": 60,
        },
            cache_tools_list = True,
            max_retry_attempts = 3
    )
    await mcp_server.connect()
    # 列出服务器所有工具
    list_tool = await mcp_server.list_tools()
    logger.debug(f"查询指定服务器:{mcp_config.mcp_base_url},获取工具列表:{list_tool}")
    try:
        # 调用工具
        result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": 5
            }
        )
        return result
    except Exception as e:
        logger.exception(f"连接mcp报错,错误信息:{str(e)}")
    finally:
        await mcp_server.cleanup()


if __name__ == "__main__":
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }
    result_state = node_web_search_mcp(test_state)
    print(result_state)

