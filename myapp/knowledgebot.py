from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from django.db.models import F
from django.utils import timezone
# At the top of your views.py or wherever KnowledgeBot is defined
from .models import QueryInsight, Department, UploadedPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from django.db.models import F
from django.utils import timezone
import os


class KnowledgeBot:
    def __init__(self, shared_vectorstore, llm):
        self.shared_vectorstore = shared_vectorstore
        self.llm = llm
        self.pdf_qa = {}
        self.memories = {}

        # The knowledge agent and output validation agent
        self.qa_prompt_template = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""Answer the question using the provided context. Be direct and concise.

Context: {context}
Chat History: {chat_history}
Question: {question}

Instructions:
1. Give a brief, focused answer (2-3 sentences maximum)
2. Only include essential information
3. Remove any redundant details
4. Skip background information unless specifically asked

Answer: """
)

        # Query validation agent prompt template
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

        # Query validation chain (agent)
        self.query_validator = LLMChain(llm=self.llm, prompt=self.query_validation_template)

        # Insight generator agent prompt template
        self.insight_generator_template = PromptTemplate(
            input_variables=["question"],
            template="""You are an insight generator agent. Your task is to analyze the given query and extract its core meaning or essence. Provide a concise response (2-4 words) that captures the fundamental concept, intent, or implication of the query. This will be used for trend analysis, predictive modeling, and identifying knowledge gaps.

            Query: {question}

            Core essence:"""
        )
        # Insight generator chain (agent)
        self.insight_generator = LLMChain(llm=self.llm, prompt=self.insight_generator_template)

    def load_pdfs(self, pdf_paths, department, user):
        self.shared_vectorstore.load_pdfs(pdf_paths, department, user)
        self.initialize_qa(department, user)

    def initialize_qa(self, department, user):
        key = f"{user.id}_{department}"
        if key not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department, user)
            if retriever:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                self.memories[key] = memory
                self.pdf_qa[key] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": self.qa_prompt_template}
                )
            else:
                pdfs = UploadedPDF.objects.filter(department__name=department, user=user)
                if pdfs.exists():
                    pdf_paths = [pdf.file.path for pdf in pdfs if os.path.exists(pdf.file.path)]
                    if pdf_paths:
                        self.load_pdfs(pdf_paths, department, user)

    def validate_query(self, question):
        validation_result = self.query_validator.run(question)
        if validation_result.strip().upper().startswith("SAFE"):
            return True, ""
        else:
            return False, validation_result.strip()

    def generate_insight(self, question, department):
        insight = self.insight_generator.run(question).strip()
        
        # Get the Department object
        try:
            dept_obj = Department.objects.get(name=department)
        except Department.DoesNotExist:
            # Handle the case where the department doesn't exist
            print(f"Department '{department}' does not exist")
            return insight

        # Update or create the insight in the database
        query_insight, created = QueryInsight.objects.get_or_create(
            department=dept_obj,  # Use the Department object instead of its name
            topic=insight,
            defaults={'frequency': 1, 'last_queried': timezone.now()}
        )
        
        if not created:
            query_insight.frequency = F('frequency') + 1
            query_insight.last_queried = timezone.now()
            query_insight.save()
        
        return insight

    def answer_question(self, question, department, user):
        is_safe, validation_message = self.validate_query(question)
        if not is_safe:
            return f"I'm sorry, but I can't process this query. {validation_message}"

        self.initialize_qa(department, user)
        key = f"{user.id}_{department}"
        if key in self.pdf_qa:
            try:
                answer = self.pdf_qa[key]({"question": question, "chat_history": self.memories[key].buffer if key in self.memories else []})
                if not answer['answer'] or answer['answer'].strip() == "":
                    return "I couldn't find a clear answer. Please rephrase your question."
                format_prompt = f"""Summarize this answer in 2-3 clear sentences maximum: {answer['answer']} Concise answer:"""
                concise_answer = self.llm.predict(format_prompt)
                return concise_answer
            except Exception as e:
                return f"An error occurred. Please try rephrasing your question."
        else:
            return "No PDFs found for this department. Please upload PDFs first."
        
    def enhance_retrieval(self, department, question, k=4):
        """
        Enhance document retrieval by using multiple search strategies
        """
        if department not in self.vectorstores:
            return []
        
        # Get documents using similarity search
        similar_docs = self.vectorstores[department].similarity_search(question, k=k)
        
        # Get documents using MMR (Maximum Marginal Relevance)
        mmr_docs = self.vectorstores[department].max_marginal_relevance_search(question, k=k)
        
        # Combine and deduplicate results
        all_docs = similar_docs + mmr_docs
        unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
        
        return unique_docs[:k]
    
    def initialize_qa(self, department, user):  # This line is correct now
        key = f"{user.id}_{department}"
        if key not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department, user)
            if retriever:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                self.memories[key] = memory
                self.pdf_qa[key] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": self.qa_prompt_template}
                )
            else:
                pdfs = UploadedPDF.objects.filter(department__name=department, user=user)
                if pdfs.exists():
                    pdf_paths = [pdf.file.path for pdf in pdfs if os.path.exists(pdf.file.path)]
                    if pdf_paths:
                        self.load_pdfs(pdf_paths, department, user)
        
    def clear_memory(self, department):
        if department in self.memories:
            self.memories[department].clear()

    def get_top_insights(self, department, limit=5):
        return QueryInsight.objects.filter(department_id=department).order_by('-frequency')[:limit]
