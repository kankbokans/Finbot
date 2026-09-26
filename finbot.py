import os
import pprint
import getpass
import bs4
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

import gradio as gr
import streamlit as st

load_dotenv()

#Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#Setup and data loading
def get_sitemap(url):
    req = Request(
        url=url,
        headers ={"User-Agent": "Mozilla/5.0"}
    )

    response = urlopen(req)
    xml = BeautifulSoup(
        response,
        "lxml-xml",
        from_encoding=response.info().get_param("charset")
    )
    return xml

def get_urls(xml, name=None,data=None, verbose = False):
    urls=[]
    for url in xml.find_all("url"):
        if xml.find("loc"):
            loc = url.findNext("loc").text
            urls.append(loc)
    return urls

#Vectorstore and Retriever
#Load the saved vector database if it exists; otherwise build and save it.
#Delete the chroma_db folder to rebuild from the latest Varsity content.
persist_directory = "chroma_db"

if os.path.isdir(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    print(f"Loaded vector database from {persist_directory}")
else:
    url = "https://zerodha.com/varsity/chapter-sitemap2.xml"
    xml= get_sitemap(url)
    urls =get_urls(xml, verbose=False)

    docs=[]
    for i, url in enumerate(urls):
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
        if i%10 == 0:
            print("Loaded document index:", i)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings(),persist_directory=persist_directory)
    print(f"Len docs: {len(docs)}, Len splits: {len(splits)}")

retriever = vectorstore.as_retriever()

#RAG Chain Setup
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
system_prompt = (
    "You are a financial assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you don't know."
    "Use three sentences maximum and keep the answer concise. "
    "If the question is not clear ask follow up questions. "
    "\n\n"
    "{context}"
    )

#History Aware Retriever
contextualize_q_system_prompt= (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)

tools = [YahooFinanceNewsTool()]
agent_test_prompt = "What is the latest news about Indian stock market like Infosys?"

agent = create_agent(llm, tools)

print("\n--- Running Agent (LangGraph) ---")
result = agent.invoke({"messages":[HumanMessage(content=agent_test_prompt)]})

#The last assistant message content:
print(result["messages"][-1].content)

#Gradio
def predict(message, history):
    """
    message: str
    history: list of dicts in OpenAI-style format when type='messages'
    e.g [{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]
    """
    history_for_llm =[]
    for item in history:
        role = item.get("role")
        content = item.get("content","")
        if role == "user":
            history_for_llm.append(HumanMessage(content=content))
        elif role == "assistant":
            history_for_llm.append(AIMessage(content=content))

    result = rag_chain.invoke({"input":message,"chat_history": history_for_llm})
    return result["answer"]

with gr.Blocks() as demo:
    gr.Markdown("# DocumentQABot")

    chatbot = gr.Chatbot(height=400, allow_tags=False)
    msg = gr.Textbox(
        placeholder= "Hi! I am your virtual assistant, how can I help you today?",
        container=False,
        scale=7,
    )

    with gr.Row():
        undo = gr.Button("Delete Previous")
        clear = gr.Button("Clear")

    chat = gr.ChatInterface(
        fn=predict,
        chatbot=chatbot,
        textbox=msg,
        examples=["What is index fund?","Where to buy stocks?"],
        title=None,   
    )

    #Clear chat
    clear.click(lambda: [], None, chatbot)

    #Undo last turn (remove last 2 messages if present: user+assistant)
    def undo_last(history):
        if not history:
            return []
        return history[:-2] if len(history)>=2 else []

    undo.click(undo_last, chatbot,chatbot)

demo.launch(share=True, debug=False)



