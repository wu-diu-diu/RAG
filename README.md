# rag-tutorial-v2-zh

- 使用Alibaba的中文embedding模型：gte-large-zh
- 使用Alibaba的qwen系列模型作为LLM
- 使用streamlit部署在本地: `steamlit run app.py`， 即可在本地`http://localhost:8501`打开
    - 如果程序部署在服务器，想在本地打开的话，需要先将本地的8501端口转发到远程服务其上的localhost:8501：`ssh -L 8501:localhost:8501 YOUR_SERVER_IP`, your_server_ip是服务器的地址。

## TODO
- [ ] 使用langchain搭建agent，使能够实现更多功能
- [√] 借助开源的前端界面部署在本地
- [ ] 为前端界面添加更多功能，比如历史会话，文件上传。
