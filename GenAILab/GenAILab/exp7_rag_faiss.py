"""
RAG Pipeline: FAISS + HuggingFace Embeddings + Ollama (Mistral)
pip install langchain langchain-community faiss-cpu sentence-transformers
Requires: ollama pull mistral
"""
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Knowledge base
docs = [
    Document(page_content="The Eiffel Tower is in Paris, France. Built in 1889."),
    Document(page_content="Python is a high-level programming language known for simplicity."),
    Document(page_content="FAISS is a Facebook library for fast similarity search on vectors."),
    Document(page_content="Machine learning lets computers learn patterns from data."),
]

# Step 1: Chunk → Embed → Store in FAISS
chunks = RecursiveCharacterTextSplitter(chunk_size=100).split_documents(docs)
db     = FAISS.from_documents(chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# Step 2: RAG chain (Retrieve → Augment → Generate)
qa = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral"),
    retriever=db.as_retriever(search_kwargs={"k": 2}),
)

questions = ["Where is the Eiffel Tower?", "What is FAISS?"]
for q in questions:
    print(f"Q: {q}\nA: {qa.run(q)}\n")
