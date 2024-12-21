import traceback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from django.conf import settings
# from KAMCO import settings
from .models import UploadedPDF, Department
from django.db import OperationalError
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from .models import UserVectorStore
from ThreadPoolExecutorPlus import ThreadPoolExecutor
import os
class SharedVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstores = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_existing_vectorstores(self, user):
        user_vectorstores = UserVectorStore.objects.filter(user=user)
        for user_vs in user_vectorstores:
            persist_directory = user_vs.vector_store_path
            if os.path.exists(persist_directory):
                self.vectorstores[f"{user.id}_{user_vs.department.name}"] = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )

    def load_pdfs(self, pdf_paths, department, user):
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            pdf_documents = loader.load()
            chunked_documents = self.text_splitter.split_documents(pdf_documents)
            documents.extend(chunked_documents)

        if documents:
            key = f"{user.id}_{department}"
            if key not in self.vectorstores:
                persist_directory = f'vectorstore_{user.id}_{department}'
                self.vectorstores[key] = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )
            self.vectorstores[key].add_documents(documents)
            self.vectorstores[key].persist()

    def get_retriever(self, department, user):
        key = f"{user.id}_{department}"
        if key in self.vectorstores:
            return self.vectorstores[key].as_retriever()
        return None

    def remove_pdf(self, pdf_name, department, user):
        key = f"{user.id}_{department}"
        if key in self.vectorstores:
            ids = self.vectorstores[key]._collection.get(where={"source": pdf_name})["ids"]
            if ids:
                self.vectorstores[key].delete(ids=ids)
                self.vectorstores[key].persist()
            else:
                print(f"No documents found for PDF: {pdf_name}") 
