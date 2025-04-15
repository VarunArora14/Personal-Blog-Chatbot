import os

import google.generativeai as genai
import instructor
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_google_genai import ChatGoogleGenerativeAI

# https://blog.streamlit.io/introducing-two-new-caching-commands-to-replace-st-cache/

st.set_page_config(
    page_title="Personal Blog Chatbot",
    page_icon="🐳",
)


@st.cache_resource
def initGeminiLLM():
    # load_dotenv()
    # GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="gemini-2.0-flash",  # model defaults to "gemini-pro"
        ),
        mode=instructor.Mode.GEMINI_JSON,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0.5, max_retries=3
    )
    return client, llm


@st.cache_resource
def loadVectorDB(folder_path="app/faissdb_1000"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vstore2 = FAISS.load_local(
        folder_path=folder_path,
        index_name="blog",
        embeddings=embedding_function,
        allow_dangerous_deserialization=True,
    )
    return vstore2


# to be used by fusion and query decomposition
if "llm" not in st.session_state:
    st.session_state["client"], st.session_state["llm"] = initGeminiLLM()

if "vstore" not in st.session_state:
    st.session_state["vstore"] = loadVectorDB()


st.write("# 🤖 Welcome to the Personal Blog Chatbot! 🌐")

st.sidebar.success("Choose Fusion or Query Decomposition to start asking")

st.markdown(
    """
    Curious how AI can help you find the most relevant information from multiple sources? You've come to the right place! This chatbot is powered by Retrieval-Augmented Generation (RAG), a cutting-edge approach where I blend smart document search with advanced AI responses.

    ✨ How it works:

    1. I gather knowledge by scraping and parsing documents from my blog posts at https://varunarora14.github.io 📄.
    2. Then, using **Query Decomposition** and **RAG Fusion**, I break down complex queries to deliver precise, well-rounded answers 🎯. You should try both and decided which one you like more😉
    3. Plus, I cite my sources—so you can trust where the information comes from 📚.
    Type in your question and let me guide you through the answers! 🌟
    
    ---
    
    """
)
