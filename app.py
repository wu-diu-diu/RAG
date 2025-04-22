import streamlit as st
import os

# 设置 Streamlit 配置
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'

from query_data import query_rag

st.set_page_config(
    page_title="RAG 问答系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG 问答系统")

# 添加侧边栏说明
with st.sidebar:
    st.markdown("""
    ### 使用说明
    1. 在输入框中输入你的问题
    2. 系统会从知识库中检索相关信息
    3. 基于检索到的信息生成答案
    
    ### 关于系统
    - 使用 Qwen 2.5 7B 模型
    - 基于 RAG (检索增强生成) 技术
    - 支持中文和英文问答
    """)


with st.form("question_form"):
    # 在表单内添加输入框
    question = st.text_input("请输入您的问题:", key="question_input")
    
    # 添加提交按钮
    submitted = st.form_submit_button("提交问题")
    
    # 如果表单被提交（无论是按回车还是点击按钮）
    if submitted:
        if question:
            # 调用RAG查询函数
            with st.spinner("正在思考中..."):
                try:
                    # 调用 RAG 查询函数
                    response = query_rag(question)
                    
                    # 显示回答
                    st.markdown("### 回答")
                    st.write(response["回答"])

                    st.markdown("### 信息源")
                    st.write(response["信息源"])
                    
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")
        else:
            st.warning("请输入问题后再提交")