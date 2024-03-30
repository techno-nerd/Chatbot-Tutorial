from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

def split_docs(documents,chunk_size=1000,chunk_overlap=100):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n","\n","."," ",""])
  docs = text_splitter.split_documents(documents)
  return docs

documents = load_docs("documents/")
docs = split_docs(documents)

embedding_model = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local("vectors")