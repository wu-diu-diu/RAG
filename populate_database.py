import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "Knowledge_data"

def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents(DATA_PATH):
    """加载所有文档，包括PDF和MD文件"""
    documents = []
    
    if os.path.isdir(DATA_PATH):
        # 加载PDF文件
        pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents.extend(pdf_loader.load())
        print(f"📚 Loaded {len(documents)} PDF documents")

        md_loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )
        # 加载MD文件
        md_docs = md_loader.load()
        documents.extend(md_docs)
        print(f"📝 Loaded {len(md_docs)} Markdown documents")
    else:
        raise NotADirectoryError(f"路径 {DATA_PATH} 不是目录")
    return documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    ## 为同属于一页的chunks按照先后顺序给定一个id，并将这个id信息添加到chunks的metadata属性中
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    """
    为chunks生成唯一ID
    对于PDF文件：source:page:chunk_index
    对于MD文件：source:chunk_index
    """

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        # 根据是否有page信息生成不同的ID格式
        if page is not None:
            current_page_id = f"{source}:{page}"
        else:
            current_page_id = source
        
        # 计算chunk ID

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        ## 在metadata字典中添加一个id键
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
