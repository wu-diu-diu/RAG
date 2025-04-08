---
title: RAG-langchain
date: 2025-04-07 16:27:32
tags: LLM
---

# RAG学习

## 数据加载

我们需要利用给定的数据形成知识库，数据的类型多种多样，langchain提供了很多工具来帮助我们加载和处理数据。比如加载PDF文档。

### 从某一个目录下加载所有PDF文档
```python
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = "data"

document_loader = PyPDFDirectoryLoader(DATA_PATH)
document = document_loader.load()
```
上面返回的document是一个列表，其中的每一个元素是一个`Document`类，查看其属性如下：
```python
document[0].__dict__

{'id': None,
 'metadata': {'producer': 'Acrobat Distiller 9.5.5 (Macintosh)',
  'creator': 'QuarkXPress 9.5.3.1',
  'creationdate': '2015-03-09T10:16:51-07:00',
  'author': 'cyrille',
  'gts_pdfxversion': 'PDF/X-3:2002',
  'moddate': '2015-05-18T11:25:26-07:00',
  'title': '[T2R] rules EN reprint 2015_TTR2 rules US',
  'trapped': '/False',
  'source': 'data/ticket_to_ride.pdf',
  'total_pages': 4,
  'page': 0,
  'page_label': '1'},
 'page_content': 'O\nn a blustery autumn evening five old friends...',
 'type': 'Document'}
```
主要有四个属性，id，metadata, page_content, type。本例中一共有两个PDF文件，共12页，document这个列表的长度也为12。所以该方法是对每页PDF提取文字，构建一个Document类并返回。

### 文档拆分

我们从PDF文件中提取出文本内容后，需要将文本拆分成句子再变成向量。文本到句子这个过程就需要使用文本切分工具，怎么切分，切多大，两个相邻切分块之前的重合文本数量是多少这些都会影响我们最终的检索效果。

`RecursiveCharacterTextSplitter`是一个用于将长文本分割成较小块的工具，通常用于自然语言处理任务中，特别是在处理嵌入和向量数据库时。代码如下：

```python
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  ## 每个文本块的最大字符数
        chunk_overlap=80,  ## 相邻文本块之间的重叠字符数，用于保持上下文的连贯性
        length_function=len, ## 用于计算文本长度的函数，默认是 len，即字符数。
        is_separator_regex=False,  ## 是否使用正则表达式来识别分隔符，这里使用默认的分隔符比如空格来切分字符
    )
## split_documents方法接受document类作为输入
## 返回还是一个包含document
chunk = text_splitter.split_documents(documents)

## 当然，还可以接受一般的字符串作为输入进行切分
text = "我们从PDF文件中提取出文本内容后，需要将文本拆分成句子再变成向量。文本到句子这个过程就需要使用文本切分工具，怎么切分，切多大，两个相邻切分块之前的重合文本数量是多少这些都会影响我们最终的检索效果。"

chunks = text_splitter.split_text(text)
```
split_documents方法返回的还是一个包含document元素的列表，只不过将原来的内容按照要求分隔的更小了。其属性包含如下：

```python
chunk[0].__dict__

{'id': None,
 'metadata': {'producer': 'Acrobat Distiller 9.5.5 (Macintosh)',
  'creator': 'QuarkXPress 9.5.3.1',
  'creationdate': '2015-03-09T10:16:51-07:00',
  'author': 'cyrille',
  'gts_pdfxversion': 'PDF/X-3:2002',
  'moddate': '2015-05-18T11:25:26-07:00',
  'title': '[T2R] rules EN reprint 2015_TTR2 rules US',
  'trapped': '/False',
  'source': 'data/ticket_to_ride.pdf',
  'total_pages': 4,
  'page': 0,
  'page_label': '1'},
 'page_content': 'O\nn a blustery autumn evening five old friends met in the backroom of one of the city’s oldest and most private clubs. Each had\ntraveled a long distance — from all corners of the world — to meet on this very specific day… October 2, 1900 — 28 years to the\nday that the London eccentric, Phileas Fogg accepted and then won a £20,000 bet that he could travel Around the World in 80 Days . \nWhen the story of Fogg’s triumphant journey filled all the newspapers of the day, the five attended University together. Inspired by\nhis impetuous gamble, and a few pints from the local pub, the group commemorated his circumnavigation with a more modest excur-\nsion and wager – a bottle of good claret to the first to make it to Le Procope in Paris.',
 'type': 'Document'}
```

### 为chunk添加id

为了提高RAG的可信度，我们需要LLM在利用RAG回答完问题后，能给出具体参考了哪个文件的哪一个段落。那么就需要我们为每一个chunk添加一个能显示上述信息的id。函数如下：
```python
def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2" 表示monopoly这个pdf的第六页中的第3个chunk
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

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
```
该函数的作用本质上是将chunk自带的source和page信息抽取出来，加一个if来判断当前chunk和上一个chunk是否属于同一页内容，属于则当前chunk的id为上一下chunk的id加1，不属于则当前chunk的id为0，表示新的一页内容。所以每个chunk的metadata中都添加了一个id键，其属性为:`"data/monopoly.pdf:6:2"`记录了该chunk来自哪篇文件，哪一页的第几个chunk。

### 添加到向量数据库中
[向量数据库Chroma极简教程](https://zhuanlan.zhihu.com/p/665715823)

LLM 的核心支持技术之一是向量嵌入。虽然计算机不能直接理解文本，但我们可以将文本表示成向量以供计算机去理解和计算。在transformer中，模型就是直接处理每个token的语义向量来理解句子的意思。注意力机制相当于LLM的大脑，我们输入文本，LLM经过思考后回答。但是LLM的知识是有限的，有时候是过时的。如果我们想要LLM对于自己不熟悉的内容也能给出回答，那么就需要给LLM一个字典，这个字典用来存储一些LLM在训练中没有见过的知识。这个字典就是向量数据库。

向量存储是专门为有效地存储和检索向量嵌入而设计的数据库。之所以需要它们，是因为像 SQL 这样的传统数据库没有针对存储和查询大型向量数据进行优化。向量存储可以使用相似性算法对相似的向量进行索引和快速搜索。它允许应用程序在给定目标向量查询的情况下查找相关向量。

`ChromaDB`是一款开源的矢量存储数据库，用于存储和检索矢量嵌入。它的主要用途是保存嵌入和元数据，以便以后由大型语言模型使用。此外，它还可用于文本数据的语义搜索引擎。

在langchain中，我们可以这样使用：第一个参数为存储的数据库的集合命名，第二个参数为本地存储的路径，第三个为embedding的函数。
```python
db = Chroma(
        collection_name="example_collection",
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
```
以上代码初始化一个向量数据库，接下来我们可以使用`add_documents`方法来将上面的chunks都添加到向量数据库中，该方法会自动将文档即document中的内容转化为嵌入向量并存储。参数如下：
- documents: 要添加的文档列表
- metadatas: 与文档关联的元数据列表，用于存储额外信息并支持过滤。
- ids: 文档的唯一标识符列表。
- embeddings: 如果提供，Chroma 将存储这些嵌入向量，而不会自行计算。

示例代码：
```python
from uuid import uuid4

from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

documents = [document_1, document_2]

uuids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=uuids)
```
其中uuid是文档对应的唯一标识符，像这样:`f22ed484-6db3-4b76-adb1-18a777426cd6`,在代码中使用 uuid4 为每个文档生成一个唯一的 ID，确保了文档的唯一性和可识别性。这在处理大量文档或分布式系统时尤为重要。从上面的代码中可以看到chunk在创建的时候是没有id的，`'id': None,`。这里我们可以用前面计算的chunk_id来作为uuid。


同时，我们可以使用`get`方法，来从当前的数据库中检索数据。参数如下：
- ids: 指定要检索的文档的唯一 ID 列表。
- where: 通过元数据字段进行过滤的条件。
- include: 指定返回的数据字段，例如 ["documents", "metadatas"]。

## 检索生成

### 相似度计算

通过以上步骤，即可用本地的文件构建一个数据库。下一步就是当用户提出query之后，我们如何在数据库中检索之后，将检索后的答案和prompt嵌入在一起来作为增强，最后输入LLM

其实从代码上来说非常简单，如下：

```python
results = db.similarity_search_with_score(query_text, k=5)

result[0] = (Document(id='data/monopoly.pdf:0:0', metadata={'creationdate': '2007-05-03T12:38:10-04:00', 'creator': 'Adobe Acrobat 7.0', 'id': 'data/monopoly.pdf:0:0', 'moddate': '2007-05-03T12:52:41-04:00', 'page': 0, 'page_label': '1', 'producer': 'Adobe Acrobat 7.0 Paper Capture Plug-in', 'source': 'data/monopoly.pdf', 'total_pages': 8}, page_content='MONOPOLY....'), 0.6146661206973629)

result[0][0].__dict__
## 可以看到document的id属性被赋予了，值就是我们刚刚计算的chunk_id
{'id': 'data/monopoly.pdf:0:0', 'metadata': {'creationdate': '2007-05-03T12:38:10-04:00', 'creator': 'Adobe Acrobat 7.0', 'id': 'data/monopoly.pdf:0:0', 'moddate': '2007-05-03T12:52:41-04:00', 'page': 0, 'page_label': '1', 'producer': 'Adobe Acrobat 7.0 Paper Capture Plug-in', 'source': 'data/monopoly.pdf', 'total_pages': 8}, 'page_content': 'MONOPOLY ....', 'type': 'Document'}
```
`similarity_search_with_score`函数返回一个包含元组的列表，每个元组包含两个元素，(文档对象，相似性分数)，相似性分数表示查询文本与文档之间的相似性，通常是一个介于 0 和 1 之间的值，值越接近 1 表示相似性越高。返回的五个文档的相似性分数是升序的。

### 检索嵌入

将检索到的前五个相似的文本的内容整合到一起，添加到用户的prompt中:

```python
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

## 将文本连起来，中间用\n\n---\n\n分隔
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
# print(prompt)

model = OllamaLLM(model="mistral")
response_text = model.invoke(prompt)

sources = [doc.metadata.get("id", None) for doc, _score in results]
formatted_response = f"Response: {response_text}\nSources: {sources}"
print(formatted_response)
return response_text
```

这里使用本地的ollama中的模型开响应。以上就是一个RAG的大致流程。本文简单学习了chroma的一些基本方法，了解了RAG的大致过程和原理。后续可能会继续学习langchain的智能体构建方法，敬请期待。

[项目参考地址](https://github.com/pixegami/rag-tutorial-v2)