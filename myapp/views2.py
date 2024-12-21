        
import re
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from .models import UploadedPDF, Department
from django.db import OperationalError
from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from .models import UserVectorStore
# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
api_key = os.getenv('GROQ_API_KEY')
# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Ensure that GROQ_API_KEY is set in the environment

# llm = ChatGroq(temperature=0, model_name='llama-3.1-70b-versatile')

def select_department(request):
    if request.method == 'POST':
        department = request.POST.get('department')
        request.session['department'] = department
        return redirect('home')
    elif request.method == 'GET':
        department = request.GET.get('department')
        if department:
            request.session['department'] = department
            return redirect('home')
    return render(request, 'departments.html')

def home(request):
    department = request.session.get('department')
    if not department:
        return redirect('select_department')
    return render(request, 'home.html', {'department': department})

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
            
# Initialize the shared vector store
shared_vectorstore = SharedVectorStore()
# At the top of your views.py or wherever KnowledgeBot is defined
from .models import QueryInsight, Department, UploadedPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from django.db.models import F
from django.utils import timezone

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
            
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from .models import QueryLog
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
import json
from datetime import datetime, timedelta



from typing import Dict, List, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from datetime import datetime

from typing import List
from datetime import datetime
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.utilities import DuckDuckGoSearchAPIWrapper
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

from dotenv import load_dotenv
import os

load_dotenv()

# Email settings
EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')

class UniversalBot:
    def __init__(self, llm, shared_vectorstore):
        self.llm = llm
        self.shared_vectorstore = shared_vectorstore
        self.conversation_histories = {}
        self.search = DuckDuckGoSearchAPIWrapper()
        self.email_draft = None
        self.email_host_user = os.getenv('EMAIL_HOST_USER')
        self.email_host_password = os.getenv('EMAIL_HOST_PASSWORD')
        self.last_response = None
        
        # New attributes for persona adaptation
        self.user_profiles = {}
        self.communication_styles = {}
        

        self.search_tool = Tool(
            name="web_search",
            func=self.search.run,
            description="Search the web for current information"
        )

    def validate_query(self, question: str) -> tuple[bool, str]:
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Validate queries based on:
                1. ALLOW general questions
                2. ALLOW sensitive topics
                3. Block explicit harm or illegal content
                Return VALID or INVALID with reason"""),
            ("human", "{question}")
        ])
        response = self.llm.invoke(validation_prompt.format(question=question))
        is_valid = "VALID" in response.content.upper()
        reason = response.content if not is_valid else ""
        return is_valid, reason

    def retrieve_context(self, question: str, department: str, user) -> List[str]:
        retriever = self.shared_vectorstore.get_retriever(department, user)
        if retriever:
            docs = retriever.get_relevant_documents(question)
            return [doc.page_content for doc in docs]
        return []

    def search_web(self, question: str) -> str:
        try:
            return self.search_tool.run(question)
        except Exception as e:
            print(f"Search error: {str(e)}")
            return ""
        
    def analyze_for_tasks(self, conversation: dict) -> Dict:
        task_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the conversation for tasks and followups."),
            ("human", "Given this conversation: {input_text}, analyze it and identify any tasks or followups needed.")
        ])
        
        try:
            # Format the conversation into a string
            input_text = f"Question: {conversation.get('question', '')} Response: {conversation.get('response', '')}"
            
            response = self.llm.invoke(task_prompt.format(input_text=input_text))
            
            # Process the response
            try:
                # Attempt to parse direct JSON response
                return json.loads(response.content)
            except json.JSONDecodeError:
                # If not JSON, analyze the text response
                content = response.content.lower()
                
                # Default structure
                task_info = {
                    "needs_followup": False,
                    "timeframe": "none",
                    "action": "none",
                    "deadline": None,
                    "priority": "low"
                }
                
                # Simple text analysis for task detection
                if any(word in content for word in ["followup", "reminder", "schedule", "deadline", "task"]):
                    task_info["needs_followup"] = True
                    
                    # Detect timeframe
                    if any(word in content for word in ["immediate", "today", "now"]):
                        task_info["timeframe"] = "immediate"
                    elif any(word in content for word in ["tomorrow", "days", "next week"]):
                        task_info["timeframe"] = "days"
                    elif any(word in content for word in ["month", "weeks", "later"]):
                        task_info["timeframe"] = "weeks"
                    
                    # Detect priority
                    if any(word in content for word in ["urgent", "asap", "immediate"]):
                        task_info["priority"] = "high"
                    elif any(word in content for word in ["soon", "next"]):
                        task_info["priority"] = "medium"
                    
                    # Extract action (simple version)
                    sentences = content.split('.')
                    for sentence in sentences:
                        if any(word in sentence for word in ["need to", "should", "must", "have to"]):
                            task_info["action"] = sentence.strip()
                            break
                
                return task_info
                
        except Exception as e:
            print(f"Task analysis error: {e}")
            return {
                "needs_followup": False,
                "timeframe": "none",
                "action": "none",
                "deadline": None,
                "priority": "low"
            }

    def schedule_task(self, user_id: str, task_info: Dict):
        if user_id not in self.scheduled_tasks:
            self.scheduled_tasks[user_id] = []
            
        task = {
            "created_at": datetime.now().isoformat(),
            "deadline": task_info.get("deadline"),
            "action": task_info.get("action"),
            "priority": task_info.get("priority"),
            "status": "pending"
        }
        
        self.scheduled_tasks[user_id].append(task)
        
        # Create reminder if needed
        if task_info.get("timeframe") == "immediate":
            self.create_reminder(user_id, task, timedelta(hours=1))
        elif task_info.get("timeframe") == "days":
            self.create_reminder(user_id, task, timedelta(days=1))
        elif task_info.get("timeframe") == "weeks":
            self.create_reminder(user_id, task, timedelta(weeks=1))

    def create_reminder(self, user_id: str, task: Dict, delay: timedelta):
        reminder_time = datetime.now() + delay
        if user_id not in self.reminders:
            self.reminders[user_id] = []
            
        reminder = {
            "time": reminder_time.isoformat(),
            "task": task,
            "sent": False
        }
        
        self.reminders[user_id].append(reminder)

    def check_reminders(self, user_id: str) -> List[Dict]:
        """Check for due reminders"""
        if user_id not in self.reminders:
            return []
            
        current_time = datetime.now()
        due_reminders = []
        
        for reminder in self.reminders[user_id]:
            reminder_time = datetime.fromisoformat(reminder["time"])
            if not reminder["sent"] and reminder_time <= current_time:
                reminder["sent"] = True
                due_reminders.append(reminder)
                
        return due_reminders

    def analyze_communication_style(self, user_history: List[dict]) -> Dict:
        """Analyze user communication patterns for personalization"""
        style_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze user communication patterns to identify preferred style."),
            ("human", "Based on this conversation history: {history}, what is their preferred communication style?")
        ])
        
        try:
            response = self.llm.invoke(style_prompt.format(history=str(user_history)))
            default_style = {
                "formality": "professional",
                "tone": "friendly",
                "preferred_length": "medium"
            }
            
            try:
                style_dict = json.loads(response.content)
                return {**default_style, **style_dict}
            except:
                return default_style
                
        except Exception as e:
            print(f"Style analysis error: {str(e)}")
            return {
                "formality": "professional",
                "tone": "friendly", 
                "preferred_length": "medium"
            }

    def update_user_profile(self, user_id: str, interaction: Dict):
        """Update user profile with new interaction data"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "interactions": [],
                "style_analysis": None
            }
            
        self.user_profiles[user_id]["interactions"].append(interaction)
        
        # Analyze style every 5 interactions
        if len(self.user_profiles[user_id]["interactions"]) % 5 == 0:
            style = self.analyze_communication_style(
                self.user_profiles[user_id]["interactions"]
            )
            self.user_profiles[user_id]["style_analysis"] = style
            self.communication_styles[user_id] = style

    def adapt_response_style(self, response: str, user_id: str) -> str:
        """Adapt response to user's preferred style"""
        if user_id not in self.communication_styles:
            return response
            
        style = self.communication_styles[user_id]
        
        style_adaptation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Adapt this response to match:
            Formality: {formality}
            Tone: {tone}
            Length: {preferred_length}
            Keep the same information but adjust the style."""),
            ("human", "{response}")
        ])
        
        try:
            adapted_response = self.llm.invoke(
                style_adaptation_prompt.format(
                    response=response,
                    **style
                )
            )
            return adapted_response.content
        except Exception as e:
            print(f"Style adaptation error: {str(e)}")
            return response

    def generate_response(self, question: str, pdf_context: List[str], web_results: str, chat_history: List[dict]) -> str:
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """Use the following to answer:
                PDF Context: {pdf_context}
                Web Results: {web_results}
                Chat History: {chat_history}
                Generate a comprehensive response."""),
            ("human", "{question}")
        ])
        
        response = self.llm.invoke(
            response_prompt.format(
                pdf_context="\n".join(pdf_context),
                web_results=web_results,
                chat_history=str(chat_history),
                question=question
            )
        )
        return response.content

    def detect_email_intent(self, question: str) -> tuple[bool, str, str]:
        """
        Enhanced email intent detection that handles more variations
        Returns: (is_email_intent, action_type, target_email)
        """
        # Common email-related keywords
        email_keywords = ['send', 'email', 'forward', 'share']
        
        question_lower = question.lower()
        
        # Check if this is an email intent
        is_email_intent = any(keyword in question_lower for keyword in email_keywords)
        
        # Extract email using regex
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        email_match = re.search(email_pattern, question)
        target_email = email_match.group(0) if email_match else None
        
        # Determine action type
        action_type = 'NONE'
        if is_email_intent:
            if 'last' in question_lower or 'previous' in question_lower:
                action_type = 'SEND_LAST'
            else:
                action_type = 'CREATE_NEW'
                
        return is_email_intent, action_type, target_email
    
    def generate_email(self, prompt: str, last_message: str = None) -> str:
        email_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a professional email based on the content.
            If last_message is provided, use it as the main content.
            Include subject line and body. 
            Format as:
            SUBJECT: <subject>
            BODY: <email body>"""),
            ("human", f"Content: {prompt}\nLast message: {last_message if last_message else ''}")
        ])
        
        response = self.llm.invoke(email_prompt.format(prompt=prompt))
        self.email_draft = response.content
        return response.content

    def send_email(self, to_email: str, user_email: str, email_password: str) -> tuple[bool, str]:
        try:
            if not self.email_draft:
                return False, "No email draft found. Please generate an email first."
            
            lines = self.email_draft.split('\n')
            subject = lines[0].replace('SUBJECT:', '').strip()
            body = '\n'.join(lines[2:])
            
            msg = MIMEMultipart()
            msg['From'] = user_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(user_email, email_password)
                server.send_message(msg)
            
            return True, "Email sent successfully!"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"

    def handle_email_and_reminder_request(self, question: str, user) -> str:
        """Enhanced handler for email and reminder requests"""
        # Extract email address
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', question)
        if not email_match:
            return "Please provide a valid email address."
        recipient_email = email_match.group(0)
        
        # Extract time interval with more variations
        time_patterns = [
            r'(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs)',
            r'after\s+(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs)',
            r'in\s+(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs)'
        ]
        
        time_match = None
        for pattern in time_patterns:
            match = re.search(pattern, question.lower())
            if match:
                time_match = match
                break
                
        if not time_match:
            return "Please specify a reminder time (e.g., '5 minutes' or '1 hour')."
            
        amount = int(time_match.group(1))
        unit = time_match.group(2)
        
        # Calculate reminder delay
        if unit.startswith(('minute', 'min')):
            reminder_delay = timedelta(minutes=amount)
        else:
            reminder_delay = timedelta(hours=amount)
        
        # Get the content to send
        content_to_send = self.last_response if self.last_response else {
            'question': '',
            'response': 'No previous content found.'
        }
        
        # Generate email content
        email_content = f"""
        SUBJECT: Meeting Notes and Discussion Summary
        
        BODY:
        Dear Colleague,
        
        Here are the recent discussion points:
        
        Question/Topic: {content_to_send.get('question', '')}
        
        Response/Notes: {content_to_send.get('response', '')}
        
        Best regards,
        AI Assistant
        """
        
        # Send email
        self.email_draft = email_content
        success, email_message = self.send_email(
            recipient_email,
            self.email_host_user,
            self.email_host_password
        )
        
        if not success:
            return f"Failed to send email: {email_message}"
            
        # Schedule reminder
        reminder_time = datetime.now() + reminder_delay
        reminder_email = f"""
        SUBJECT: Reminder: Follow up on Previous Discussion
        
        BODY:
        This is your reminder to follow up on the discussion notes sent to {recipient_email}.
        
        Original Content:
        {content_to_send.get('question', '')}
        
        {content_to_send.get('response', '')}
        
        Best regards,
        AI Assistant
        """
        
        self.schedule_reminder_email(
            self.email_host_user,
            self.email_host_user,
            reminder_email,
            reminder_time
        )
        
        return f"Email sent successfully to {recipient_email}! A reminder will be sent to you in {amount} {unit}."

    def schedule_reminder_email(self, to_email: str, from_email: str, email_content: str, reminder_time: datetime):
        """Schedule a reminder email to be sent at a specific time"""
        if not hasattr(self, 'scheduled_reminder_emails'):
            self.scheduled_reminder_emails = []
        
        self.scheduled_reminder_emails.append({
            'to_email': to_email,
            'from_email': from_email,
            'content': email_content,
            'scheduled_time': reminder_time,
            'sent': False
        })

    def check_and_send_reminder_emails(self):
        """Check and send any due reminder emails"""
        if not hasattr(self, 'scheduled_reminder_emails'):
            return
        
        current_time = datetime.now()
        
        for reminder in self.scheduled_reminder_emails:
            if not reminder['sent'] and current_time >= reminder['scheduled_time']:
                self.email_draft = reminder['content']
                success, _ = self.send_email(
                    reminder['to_email'],
                    reminder['from_email'],
                    self.email_host_password
                )
                if success:
                    reminder['sent'] = True
    
    def log_interaction(self, department: str, question: str, response: str):
        try:
            # Get or create default department
            from myapp.models import Department  # Add this import at the top of file
            default_dept, _ = Department.objects.get_or_create(name="Default")
            
            QueryLog.objects.create(
                department=default_dept,
                query=question,
                response=response
            )
        except Exception as e:
            print(f"Logging error: {str(e)}")

    def get_memory_key(self, user_id: str, department: str = "default") -> str:
        return f"{user_id}_{department}"
    
    def get_conversation_history(self, user_id: str, department: str = "default") -> List[dict]:
        key = self.get_memory_key(str(user_id), department)
        return self.conversation_histories.get(key, [])

    def add_to_history(self, user_id: str, department: str, question: str, answer: str):
        key = self.get_memory_key(user_id, department)
        if key not in self.conversation_histories:
            self.conversation_histories[key] = []
        
        self.conversation_histories[key].append({
            'question': question,
            'answer': answer,  # Using 'answer' consistently
            'timestamp': datetime.now().isoformat()
        })
        
    

    def answer_question(self, question: str, department: str, user) -> str:
        """
        Enhanced answer_question method that can email any last response
        """
        try:
            user_id = str(user.id)
            question_lower = question.lower()
            
            # First check if this is an email request
            email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
            email_match = re.search(email_pattern, question)
            
            # If it's an email request and we have a match
            if ('send' in question_lower or 'email' in question_lower) and email_match:
                recipient_email = email_match.group(0)
                
                # Check if we have a last response
                if not self.last_response:
                    return "No previous content found to send. Please generate some content first."
                    
                # Prepare email content from last response
                email_content = f"""SUBJECT: {self.last_response.get('question', 'AI Assistant Response')[:50]}...

    BODY:
    Hello,

    Here is the content you requested:

    {self.last_response.get('response', '')}

    Best regards,
    AI Assistant"""

                self.email_draft = email_content
                
                # If it includes a reminder
                if 'remind' in question_lower:
                    # Extract time for reminder
                    time_pattern = r'(\d+)\s*(minute|minutes|min|mins|hour|hours|hr|hrs)'
                    time_match = re.search(time_pattern, question_lower)
                    
                    if time_match:
                        amount = int(time_match.group(1))
                        unit = time_match.group(2)
                        
                        # Calculate reminder delay
                        reminder_delay = timedelta(
                            minutes=amount if unit.startswith(('minute', 'min')) 
                            else amount * 60
                        )
                        
                        # Send the email
                        success, message = self.send_email(
                            recipient_email,
                            self.email_host_user,
                            self.email_host_password
                        )
                        
                        if not success:
                            return f"Failed to send email: {message}"
                            
                        # Schedule reminder
                        reminder_time = datetime.now() + reminder_delay
                        reminder_content = f"""SUBJECT: Reminder: Follow up on Previous Content

    BODY:
    This is your reminder about the content sent to {recipient_email}.

    Original Content:
    {self.last_response.get('response', '')}

    Best regards,
    AI Assistant"""

                        self.schedule_reminder_email(
                            self.email_host_user,
                            self.email_host_user,
                            reminder_content,
                            reminder_time
                        )
                        
                        return f"Email sent successfully to {recipient_email}! A reminder will be sent in {amount} {unit}."
                
                # Just send email without reminder
                success, message = self.send_email(
                    recipient_email,
                    self.email_host_user,
                    self.email_host_password
                )
                
                return f"Email status: {message}"

            # Regular question handling (non-email requests)
            is_valid, reason = self.validate_query(question)
            if not is_valid:
                return f"I cannot process this query: {reason}"

            # Get context and generate response
            pdf_context = self.retrieve_context(question, department, user)
            web_results = self.search_web(question)
            chat_history = self.get_conversation_history(user_id, department)
            
            response = self.generate_response(
                question=question,
                pdf_context=pdf_context,
                web_results=web_results,
                chat_history=chat_history
            )

            # Store the response
            self.last_response = {
                'question': question,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }

            # Update user profile and adapt response
            try:
                self.update_user_profile(user_id, {
                    'question': question,
                    'response': response,
                    'timestamp': datetime.now().isoformat()
                })
                adapted_response = self.adapt_response_style(response, user_id)
            except Exception as e:
                print(f"Profile update error: {str(e)}")
                adapted_response = response

            # Log and store
            self.log_interaction(department, question, adapted_response)
            self.add_to_history(user_id, department, question, adapted_response)

            return adapted_response
            
        except Exception as e:
            print(f"Critical error in answer_question: {str(e)}")
            return "I encountered an unexpected error. Please try again or contact support if the issue persists."



  

    def handle_email_action(self, question: str, user) -> str:
        # Extract email address from question
        import re
        patterns = [
            r'[\w\.-]+@[\w\.-]+\.\w+',
            r'to\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'send\s+to\s+([\w\.-]+@[\w\.-]+\.\w+)',
            r'forward\s+to\s+([\w\.-]+@[\w\.-]+\.\w+)',
        ]
        
        email_address = None
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                email_address = match.group(1) if len(match.groups()) > 0 else match.group(0)
                break
                
        if not email_address:
            return "I couldn't find an email address in your request. Please specify the email address you want to send to."

        if not self.last_response:
            return "There's no recent conversation to send. Please ask a question first."

        # Directly format the last response as email
        email_content = f"""SUBJECT: {self.last_response.get('question', 'AI Assistant Response')[:50]}...
    BODY: Hello,

    Here is the content you requested:

    {self.last_response.get('response', '')}

    Best regards,
    AI Assistant"""

        self.email_draft = email_content
        user_email = os.getenv('EMAIL_HOST_USER')
        email_password = os.getenv('EMAIL_HOST_PASSWORD')
        
        success, message = self.send_email(email_address, user_email, email_password)
        return message
    
    def format_conversation_as_email(self, last_response: dict) -> str:
        """Format the conversation into an email structure"""
        try:
            # Extract the main content
            question = last_response.get('question', 'No question available')
            response = last_response.get('response', 'No response available')
            
            # Create a clean subject line
            subject = question[:50] + "..." if len(question) > 50 else question
            
            # Format the email
            email_content = f"""SUBJECT: {subject}

    BODY:
    Hello,

    Here is the information you requested:

    Original Question/Topic:
    {question}

    Response/Content:
    {response}

    Best regards,
    AI Assistant"""

            return email_content
        except Exception as e:
            print(f"Error formatting email: {str(e)}")
            return """SUBJECT: AI Assistant Response
            
    BODY:
    There was an error formatting the content. Please try again."""

    def send_email(self, to_email: str, user_email: str, email_password: str) -> tuple[bool, str]:
        """Send email with better error handling and validation"""
        try:
            # Validate email addresses
            if not all([to_email, user_email, email_password]):
                return False, "Missing email configuration. Please check your settings."

            if not self.email_draft:
                return False, "No content found to send. Please generate content first."

            # Parse email content
            lines = self.email_draft.split('\n')
            subject = next((line.replace('SUBJECT:', '').strip() 
                        for line in lines if line.strip().startswith('SUBJECT:')), 
                        'AI Assistant Response')
            
            body_start = next((i for i, line in enumerate(lines) 
                            if line.strip().startswith('BODY:')), -1)
            
            if body_start == -1:
                body = '\n'.join(lines[1:])  # Use everything after subject if no BODY: marker
            else:
                body = '\n'.join(lines[body_start + 1:])

            # Create message
            msg = MIMEMultipart()
            msg['From'] = user_email
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body.strip(), 'plain'))

            # Send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(user_email, email_password)
                server.send_message(msg)

            return True, f"Email sent successfully to {to_email}"

        except smtplib.SMTPAuthenticationError:
            return False, "Email authentication failed. Please check your email credentials."
        except smtplib.SMTPException as e:
            return False, f"SMTP error occurred: {str(e)}"
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
        
    def clear_memory(self, user_id: str, department: str):
        key = self.get_memory_key(user_id, department)
        if key in self.conversation_histories:
            self.conversation_histories[key] = []
            
    
            
            
            
# At the top of your views.py file, after imports
# Initialize the shared vector store
shared_vectorstore = SharedVectorStore()

# Initialize the bots
knowledge_bot = KnowledgeBot(shared_vectorstore, llm)
universal_bot = UniversalBot(llm, shared_vectorstore)
from django.contrib.auth.decorators import login_required
@login_required
@csrf_exempt
def knowledge_agent(request):
    department = request.session.get('department')
    if not department:
        return redirect('select_department')

    # Initialize the QA system for the current department and user
    knowledge_bot.initialize_qa(department, request.user)

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'upload_pdfs':
            pdf_files = request.FILES.getlist('pdf_files')
            loaded_pdfs = []
            pdf_paths = []
            for pdf_file in pdf_files:
                if UploadedPDF.objects.filter(name=pdf_file.name, department__name=department, user=request.user).exists():
                    return JsonResponse({'status': f'A PDF with the name {pdf_file.name} already exists in this department for this user'}, status=400)
                file_name = default_storage.save(f'pdfs/{request.user.id}/{department}/{pdf_file.name}', ContentFile(pdf_file.read()))
                pdf_path = default_storage.path(file_name)
                pdf_paths.append(pdf_path)
                loaded_pdfs.append(pdf_file.name)
                dept_obj, _ = Department.objects.get_or_create(name=department)
                UploadedPDF.objects.create(file=file_name, name=pdf_file.name, department=dept_obj, user=request.user)
            knowledge_bot.load_pdfs(pdf_paths, department, request.user)
            return JsonResponse({'status': f'{len(loaded_pdfs)} PDFs uploaded successfully', 'loaded_pdfs': loaded_pdfs})
        elif action == 'get_pdfs':
            pdfs = UploadedPDF.objects.filter(department__name=department, user=request.user)
            return JsonResponse({'loaded_pdfs': [pdf.name for pdf in pdfs]})
        elif action == 'delete_pdf':
            pdf_name = request.POST.get('pdf_name')
            try:
                pdfs = UploadedPDF.objects.filter(name=pdf_name, department__name=department, user=request.user)
                for pdf in pdfs:
                    if default_storage.exists(pdf.file.name):
                        default_storage.delete(pdf.file.name)
                    shared_vectorstore.remove_pdf(pdf_name, department, request.user)
                    pdf.delete()
                return JsonResponse({'status': f'All instances of {pdf_name} deleted successfully'})
            except Exception as e:
                print(f"Error deleting PDF: {str(e)}")
                return JsonResponse({'status': f'Error deleting PDF: {str(e)}'}, status=500)
        elif action == 'ask_question':
            question = request.POST.get('question')
            response = knowledge_bot.answer_question(question, department, request.user)
            return JsonResponse({'response': response})

    return render(request, 'knowledge_agent.html', {'department': department})

# View function remains the same
@login_required
@csrf_exempt
def universal_agent(request):
    department = request.session.get('department')
    if not department:
        return redirect('select_department')

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'ask_question':
            question = request.POST.get('question')
            response = universal_bot.answer_question(question, "default", request.user)
            return JsonResponse({'response': response})

    return render(request, 'universal_agent.html', {'department': department})


def view_pdf(request, pdf_name):
    try:
        pdf = UploadedPDF.objects.get(name=pdf_name)
        return FileResponse(pdf.file, content_type='application/pdf')
    except UploadedPDF.DoesNotExist:
        return JsonResponse({'status': 'PDF not found'}, status=404)

def clean_pdf_database():
    pdfs = UploadedPDF.objects.all()
    deleted_count = 0
    for pdf in pdfs:
        if not os.path.exists(pdf.file.path):
            print(f"Deleting database entry for non-existent file: {pdf.name}")
            pdf.delete()
            deleted_count += 1
    print(f"Cleaned up {deleted_count} database entries for non-existent PDFs")

def load_all_pdfs_for_universal_bot():
    try:
        # Clean up the database first
        clean_pdf_database()
        # Now load the PDFs that actually exist
        pdfs = UploadedPDF.objects.all()
        pdf_paths = []
        for pdf in pdfs:
            if os.path.exists(pdf.file.path):
                pdf_paths.append(pdf.file.path)
            else:
                print(f"Warning: File {pdf.file.path} does not exist, but database entry remains.")
        print(f"Found {len(pdf_paths)} valid PDFs")
        if pdf_paths:
            universal_bot.load_pdfs(pdf_paths)
        else:
            print("No valid PDFs found to load")
    except OperationalError as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
        
from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import render
from .models import QueryInsight, Department
import json
from django.views.decorators.http import require_http_methods

#using insight agent using another LLM
@require_http_methods(["GET"])
def insight_analysis(request):
    try:
        # Get all departments from the database
        departments = Department.objects.all()
        
        department_insights = []
        
        # Collect data for each department
        for dept in departments:
            insights = QueryInsight.objects.filter(department=dept).values(
                'topic', 'frequency', 'last_queried'
            )
            insights_data = list(insights)
            
            # Prepare prompt for this department
            prompt = f"""Analyze the following query insight data for {dept.name} Department and provide:
            1. Trend Analysis: Identify and describe at least 3 significant trends in the data.
            2. Knowledge Gaps and Research Suggestions: Identify at least 3 areas where there are knowledge gaps and suggest specific research to address these gaps.
            

            Data:
            {json.dumps(insights_data, default=str)}
            """

            try:
                analysis = llm.predict(prompt)
            except Exception as e:
                analysis = f"Error generating analysis: {str(e)}"

            department_insights.append({
                'id': dept.id,
                'name': dept.name,
                'insights_data': insights_data,
                'analysis': analysis
            })

        context = {
            'department_insights': department_insights
        }
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse(context)
        
        return render(request, 'insight_analysis.html', context)

    except Exception as e:
        error_message = f"An error occurred during analysis: {str(e)}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'error': error_message}, status=500)
        return render(request, 'insight_analysis.html', {'error': error_message}, status=500)   
    
    
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

def login_view(request):
    if request.user.is_authenticated:
        return redirect('select_department')
    
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('select_department')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('select_department')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect('login')


    
    