
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
from django.contrib.auth.models import User
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
        self.load_existing_vectorstores()

    def load_existing_vectorstores(self):
        users = User.objects.all()
        for user in users:
            persist_directory = f'vectorstore_{user.username}'
            if os.path.exists(persist_directory):
                self.vectorstores[user.username] = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=persist_directory
                )

    def load_pdfs(self, pdf_paths, user):
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            pdf_documents = loader.load()
            chunked_documents = self.text_splitter.split_documents(pdf_documents)
            documents.extend(chunked_documents)
        if documents:
            if user.username not in self.vectorstores:
                self.vectorstores[user.username] = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=f'vectorstore_{user.username}'
                )
            self.vectorstores[user.username].add_documents(documents)
            self.vectorstores[user.username].persist()

    def get_retriever(self, user):
        if user.username in self.vectorstores:
            return self.vectorstores[user.username].as_retriever()
        return None

    def remove_pdf(self, pdf_name, user):
        if user.username in self.vectorstores:
            ids = self.vectorstores[user.username]._collection.get(where={"source": pdf_name})["ids"]
            if ids:
                self.vectorstores[user.username].delete(ids=ids)
                self.vectorstores[user.username].persist()
            else:
                print(f"No documents found for PDF: {pdf_name}")     
            
# Initialize the shared vector store
shared_vectorstore = SharedVectorStore()

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

    def load_pdfs(self, pdf_paths, department):
        self.shared_vectorstore.load_pdfs(pdf_paths, department)
        self.initialize_qa(department)

    def initialize_qa(self, department):
        if department not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department)
            if retriever:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                self.memories[department] = memory
                self.pdf_qa[department] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": self.qa_prompt_template}
                )
            else:
                # If no retriever exists, check if there are PDFs in the database
                pdfs = UploadedPDF.objects.filter(department__name=department)
                if pdfs.exists():
                    # If PDFs exist, load them into the vector store
                    pdf_paths = [pdf.file.path for pdf in pdfs if os.path.exists(pdf.file.path)]
                    if pdf_paths:
                        self.load_pdfs(pdf_paths, department)
                        retriever = self.shared_vectorstore.get_retriever(department)
                        if retriever:
                            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                            self.memories[department] = memory
                            self.pdf_qa[department] = ConversationalRetrievalChain.from_llm(
                                self.llm,
                                retriever=retriever,
                                memory=memory,
                                combine_docs_chain_kwargs={"prompt": self.qa_prompt_template}
                            )

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

    def answer_question(self, question, department):
        is_safe, validation_message = self.validate_query(question)
        if not is_safe:
            return f"I'm sorry, but I can't process this query. {validation_message}"

        self.initialize_qa(department)
        if department in self.pdf_qa:
            try:
                # Get initial response
                answer = self.pdf_qa[department]({
                    "question": question,
                    "chat_history": self.memories[department].buffer if department in self.memories else []
                })

                if not answer['answer'] or answer['answer'].strip() == "":
                    return "I couldn't find a clear answer. Please rephrase your question."

                # Format response to be concise
                format_prompt = f"""Summarize this answer in 2-3 clear sentences maximum:
                {answer['answer']}
                
                Concise answer:"""
                
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
    
    def initialize_qa(self, department):
        if department not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department)
            if retriever:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key='answer'
                )
                self.memories[department] = memory
                
                self.pdf_qa[department] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={
                        "prompt": self.qa_prompt_template
                    },
                    chain_type="stuff",  # Use 'stuff' method for better context handling
                    verbose=True
                )
        
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

    def load_vectorstore(self, department):
        if department not in self.pdf_qa:
            retriever = self.shared_vectorstore.get_retriever(department)
            if retriever:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                self.memories[department] = memory
                self.pdf_qa[department] = ConversationalRetrievalChain.from_llm(
                    self.llm,
                    retriever=retriever,
                    memory=memory
                )
            else:
                pdfs = UploadedPDF.objects.filter(department__name=department)
                if pdfs.exists():
                    pdf_paths = [pdf.file.path for pdf in pdfs if os.path.exists(pdf.file.path)]
                    if pdf_paths:
                        self.shared_vectorstore.load_pdfs(pdf_paths, department)
                        retriever = self.shared_vectorstore.get_retriever(department)
                        if retriever:
                            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                            self.memories[department] = memory
                            self.pdf_qa[department] = ConversationalRetrievalChain.from_llm(
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

    def answer_question(self, question, department):
        # Existing validation code
        is_safe, validation_message = self.validate_query(question)
        if not is_safe:
            return f"I'm sorry, but I can't process this query. {validation_message}"

        self.load_vectorstore(department)
        
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
    # Initialize the QA system for the current user
    knowledge_bot.initialize_qa(request.user)

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'upload_pdfs':
            pdf_files = request.FILES.getlist('pdf_files')
            loaded_pdfs = []
            pdf_paths = []
            for pdf_file in pdf_files:
                if UploadedPDF.objects.filter(name=pdf_file.name, user=request.user).exists():
                    return JsonResponse({'status': f'A PDF with the name {pdf_file.name} already exists for this user'}, status=400)
                file_name = default_storage.save(f'pdfs/{request.user.username}/{pdf_file.name}', ContentFile(pdf_file.read()))
                pdf_path = default_storage.path(file_name)
                pdf_paths.append(pdf_path)
                loaded_pdfs.append(pdf_file.name)
                UploadedPDF.objects.create(file=file_name, name=pdf_file.name, user=request.user)
            knowledge_bot.load_pdfs(pdf_paths, request.user)
            return JsonResponse({'status': f'{len(loaded_pdfs)} PDFs uploaded successfully', 'loaded_pdfs': loaded_pdfs})

        elif action == 'get_pdfs':
            pdfs = UploadedPDF.objects.filter(user=request.user)
            return JsonResponse({'loaded_pdfs': [pdf.name for pdf in pdfs]})

        elif action == 'delete_pdf':
            pdf_name = request.POST.get('pdf_name')
            try:
                pdfs = UploadedPDF.objects.filter(name=pdf_name, user=request.user)
                for pdf in pdfs:
                    if default_storage.exists(pdf.file.name):
                        default_storage.delete(pdf.file.name)
                    shared_vectorstore.remove_pdf(pdf_name, request.user)
                    pdf.delete()
                return JsonResponse({'status': f'All instances of {pdf_name} deleted successfully'})
            except Exception as e:
                print(f"Error deleting PDF: {str(e)}")  # Log the error
                return JsonResponse({'status': f'Error deleting PDF: {str(e)}'}, status=500)

        elif action == 'ask_question':
            question = request.POST.get('question')
            response = knowledge_bot.answer_question(question, request.user)
            
            # Log the query
            try:
                QueryLog.objects.create(
                    user=request.user,
                    query=question,
                    response=response
                )
            except Exception as e:
                print(f"Error logging query: {str(e)}")  # Log the error
            
            return JsonResponse({'response': response})

    # For GET requests, render the knowledge agent page
    return render(request, 'knowledge_agent.html', {'user': request.user})

@csrf_exempt
def universal_agent(request):
    department = request.session.get('department')
    if not department:
        return redirect('select_department')

    # Load the vector store for the current department
    universal_bot.load_vectorstore(department)

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'ask_question':
            question = request.POST.get('question')
            response = universal_bot.answer_question(question, department)
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

from django.contrib.auth import authenticate, login
from django.shortcuts import render, redirect

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('knowledge_agent')  # Redirect to your main view
        else:
            # Return an 'invalid login' error message
            return render(request, 'login.html', {'error': 'Invalid credentials'})
    else:
        return render(request, 'login.html')

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


    
    
