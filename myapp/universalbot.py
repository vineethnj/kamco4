# Import required libraries
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from django.conf import settings
from django.db import OperationalError
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from ThreadPoolExecutorPlus import ThreadPoolExecutor
import os
import re
import random

# Import models
from .models import (
    QueryLog,
    UploadedPDF,
    Department,
    UserVectorStore
)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class UniversalBot:
    def __init__(self, llm, shared_vectorstore):
        self.llm = llm
        self.shared_vectorstore = shared_vectorstore
        self.pdf_qa = {}
        self.memories = {}
        self.search = DuckDuckGoSearchAPIWrapper()
        self.search_tool = Tool(
            name="DuckDuckGo Search",
            func=self.search.run,
            description="Useful for when you need to answer questions about current events or information not found in the PDFs"
        )
        
        self.logging_template = PromptTemplate(
            input_variables=["question", "answer", "department"],
            template="""
            As a logging agent, analyze the following query and response:
            Department: {department}
            Query: {question}
            Response: {answer}
            
            Store this interaction for future analysis and improvement of the system.
            """
        )
        
        self.logging_chain = LLMChain(llm=self.llm, prompt=self.logging_template)

        # New query validation prompt template
        self.query_validation_template = PromptTemplate(
            input_variables=["question"],
            template="""You are a very liberal query validation agent. Your primary goal is to allow almost all queries through, only blocking those that are explicitly and directly harmful.

        Query: {question}

        Validation Rules:
        1. ALLOW all general questions, even if controversial
        2. ALLOW questions about sensitive topics
        3. ALLOW questions containing strong language
        4. ALLOW business, technical, and personal questions
        5. Only mark as UNSAFE if the query explicitly:
        - Promotes direct violence or terrorism
        - Contains instructions for illegal weapons
        - Requests help with serious crimes

        Response Guidelines:
        - Default to 'SAFE' unless query is clearly dangerous
        - Do not block queries just because they seem inappropriate
        - Give users the benefit of the doubt
        - Allow controversial or adult topics
        - Focus only on preventing clear harm

        Respond with only 'SAFE' or 'UNSAFE: [brief reason]'
        """
        )
        
        # Create the query validation chain
        self.query_validator = LLMChain(llm=self.llm, prompt=self.query_validation_template)

    def load_vectorstore(self, department, user):
        key = f"{user.id}_{department}"
        if key not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department, user)
            if retriever:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                self.memories[key] = memory
                self.pdf_qa[key] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory
                )
            else:
                pdfs = UploadedPDF.objects.filter(department__name=department, user=user)
                if pdfs.exists():
                    pdf_paths = [pdf.file.path for pdf in pdfs if os.path.exists(pdf.file.path)]
                    if pdf_paths:
                        self.shared_vectorstore.load_pdfs(pdf_paths, department, user)
                        retriever = self.shared_vectorstore.get_retriever(department, user)
                        if retriever:
                            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                            self.memories[key] = memory
                            self.pdf_qa[key] = ConversationalRetrievalChain.from_llm(
                                self.llm,
                                retriever=retriever,
                                memory=memory
                            )

    def log_interaction(self, question, answer, department):
        """
        Log the query and response to the database
        """
        try:
            dept_obj = Department.objects.get(name=department)
            QueryLog.objects.create(
                department=dept_obj,
                query=question,
                response=answer
            )
            
            # Run the logging chain for analysis
            self.logging_chain.run(
                question=question,
                answer=answer,
                department=department
            )
        except Exception as e:
            print(f"Error logging interaction: {str(e)}")
    
    def validate_query(self, question):
        validation_result = self.query_validator.run(question)
        if validation_result.strip().upper().startswith("SAFE"):
            return True, ""
        else:
            return False, validation_result.strip()

    def answer_question(self, question, department,user):
        # Existing validation code
        is_safe, validation_message = self.validate_query(question)
        if not is_safe:
            return f"I'm sorry, but I can't process this query. {validation_message}"


        self.load_vectorstore(department, user)
        key = f"{user.id}_{department}"
        
        # Get PDF information
        pdf_info = ""
        if department in self.pdf_qa:
            pdf_result = self.pdf_qa[department]({"question": f"Provide any relevant information from the PDFs for this task or question: {question}"})
            pdf_info = pdf_result['answer']

        # Get web search results
        web_info = self.search_tool.run(question)

        # Combine information and generate response
        combined_prompt = f""" 
        You are a versatile AI assistant with access to various information sources. 
        Use the following information to complete the user's task or answer their question:
        1. Relevant PDF Information: {pdf_info}
        2. Web Search Results: {web_info}
        3. Chat History: {self.memories[department].buffer if department in self.memories else "No previous conversation"}
        
        Remember to incorporate specific details from the PDF information when relevant to the task.
        If the task requires creating content (like emails, proposals, or messages), 
        use the PDF information to make the content more specific and accurate.
        
        User's task/question: {question}
        Your response:
        """

        final_answer = self.llm.predict(combined_prompt)

        # Update memory
        if department in self.memories:
            self.memories[department].save_context(
                {"input": question},
                {"output": final_answer}
            )

        # Log the interaction
        self.log_interaction(question, final_answer, department)

        return final_answer

    def clear_memory(self, department):
        if department in self.memories:
            self.memories[department].clear()
                            
