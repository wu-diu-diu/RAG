import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    # query_text = "什么是垄断？"
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # 分数越低表示相关性越高
    results = db.similarity_search_with_score(query_text, k=5)
    
    # 设置相似度阈值
    SIMILARITY_THRESHOLD = 1.0
    
    # 检查是否有足够相似的文档
    # 五个结果中，只要有一个的结果的分值小于1.0，即表示在本地找到了相关文档
    has_relevant_docs = any(score < SIMILARITY_THRESHOLD for _, score in results)
    
    if has_relevant_docs:
        # 如果有相关文档，使用上下文回答
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        sources = [doc.metadata.get("id", None) for doc, _score in results]
    else:
        # 如果没有相关文档，直接使用问题
        prompt = query_text
        sources = []
    
    model = OllamaLLM(model="qwen2.5:7b")
    response_text = model.invoke(prompt)
    
    if sources:
        formatted_response = f"Response: {response_text} \n Sources: {sources}"
        dispaly_response = {"回答": response_text, "信息源": sources}
    else:
        formatted_response = f"Response: {response_text} \n Note: No relevant documents found in the knowledge base."
        dispaly_response = {"回答": response_text, "信息源": "未在本地检索到相关信息"}
    
    print(formatted_response)
    return dispaly_response


if __name__ == "__main__":
    main()
