from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


model = ChatOpenAI(model='deepseek-chat', temperature=0)
msg = HumanMessage(content="你好，请给我讲一个关于研究生的地狱笑话？", name="punchy")
messages = [msg]
messages = model.invoke(messages)
messages = [messages]

for m in messages:
    m.pretty_print()