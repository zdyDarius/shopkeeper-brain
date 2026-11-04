## 生成目标
【核心目标】 生成一个用于ragas 评估的评估测试集 
【已知背景】 待评估的资料文档已经放在了 output/upload/20261103/3c377fe1944248c3ae32df6edb1543e2/hak180产品安全手册/hak180产品安全手册_new.md 
【具体要求】
1 从资料文档中提取10组问题和参考答案
2 将提取的问题和参考答案保存在 eval\qa.csv
3 qa.csv 文件分两列 question 和 ground_truth 
4 qa.csv 文件格式 csv 编码集 utf-8 bom
5 请将"HAK 180 烫金机:" 作为问题的前缀



##生成评估测试程序
【核心目标】 生成一个基于ragas的评估测试程序 【已知背景】 
1 已经准备好了评估测试集放在 eval\qa.csv 
2 需要评估的rag流程的入口在 processor\query_process\main_graph.py 
3 需要的上下文从state中的 reranked_docs 中获取 
4 需要用到的embedding客户端工具和llm客户端工具从 utils/lm/embedding_utils.py 和 utils/lm/llm_client_util.py 中获取 
【具体要求】
1 评估生成五大指标：Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness 
2 生成的评估结果 保存在 eval\result.csv 中, 9列，列名分别为： question、contexts、answer、ground_truth、 Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness
 3 生成代码，要求每个函数都要有中文注释 4 要求每个核心步骤函数的命名要以 step_1_xx、 step_2_xx 开头。 
 5 程序保存在eval目录下 eval.py