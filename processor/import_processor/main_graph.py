from langgraph.graph import StateGraph
import json
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_document_split import node_document_split
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.nodes.node_import_milvus import node_import_milvus
from processor.import_processor.nodes.node_item_name_recognition import node_item_name_recognition
from processor.import_processor.nodes.node_md_img import node_md_img
from processor.import_processor.nodes.node_pdf_to_md import node_pdf_to_md

from langgraph.constants import START, END

from common.logging.logger import logger
from processor.import_processor.state import ImportGraphState, create_default_state



workflow = StateGraph(ImportGraphState)

# 入口节点
workflow.add_node(node_entry)
# pdf 转 md 节点
workflow.add_node(node_pdf_to_md)
# md 文档 图片处理节点
workflow.add_node(node_md_img)
# 文本切割节点
workflow.add_node(node_document_split)
# 主体识别节点
workflow.add_node(node_item_name_recognition)
# embedding 向量转化节点
workflow.add_node(node_bge_embedding)
# 向量存入向量数据库节点
workflow.add_node(node_import_milvus)


# 添加边
workflow.set_entry_point('node_entry')
# 路由函数
def after_node_entry(state: ImportGraphState):
    if state.get('is_pdf_read_enabled'):
        return 'node_pdf_to_md'

    elif state.get('is_md_read_enabled'):
        return 'node_md_img'
    else:
        return END
# 条件边
workflow.add_conditional_edges(
    source='node_entry',
    path=after_node_entry,
    path_map={
        'node_pdf_to_md': 'node_pdf_to_md',
        'node_md_img': 'node_md_img',
        END: END
    }
)
workflow.add_edge('node_pdf_to_md', 'node_md_img')
workflow.add_edge('node_md_img', 'node_document_split')
workflow.add_edge( 'node_document_split', 'node_item_name_recognition')
workflow.add_edge( 'node_item_name_recognition', 'node_bge_embedding')
workflow.add_edge("node_bge_embedding","node_import_milvus")
workflow.add_edge("node_import_milvus", END)
# 编译
kb_import_app = workflow.compile()


if __name__ == "__main__":
    logger.info("===== 开始测试 =====")

    initial_state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    final_state = None

    # 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
    # for event in kb_import_app.stream(initial_state):
    #     for key, value in event.items():
    #         logger.info(f"节点: {key}")
    #         final_state = value

    kb_import_app.invoke(initial_state)
    # 格式化输出最终状态
    logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

    logger.info("图结构:")
    # uv add grandalf
    kb_import_app.get_graph().print_ascii()

    logger.info("===== 测试结束 =====")










