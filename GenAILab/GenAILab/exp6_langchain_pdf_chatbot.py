"""
PDF Chatbot with LangChain + Streamlit + Ollama
pip install streamlit langchain langchain-community faiss-cpu sentence-transformers pypdf
Run: streamlit run exp6_langchain_pdf_chatbot.py
Requires: ollama running  →  ollama pull mistral
"""
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

st.title("PDF Q&A Chatbot")

pdf = st.file_uploader("Upload PDF", type="pdf")
if pdf:
    open("tmp.pdf","wb").write(pdf.read())
    docs   = PyPDFLoader("tmp.pdf").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db     = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    qa     = RetrievalQA.from_chain_type(Ollama(model="mistral"), retriever=db.as_retriever())

    q = st.text_input("Ask a question:")
    if q:
        st.write(qa.run(q))
