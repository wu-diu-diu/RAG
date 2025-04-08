# from langchain_ollama import OllamaEmbeddings


# def get_embedding_function():
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # 使用阿里巴巴的GTE模型
    model_name = "thenlper/gte-large-zh"  # 中文专用大模型
    model_kwargs = {'device': 'cuda'}  # 使用GPU加速
    encode_kwargs = {'normalize_embeddings': True}  # 归一化向量
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )