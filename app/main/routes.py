from flask import render_template, redirect, url_for, request, flash, send_from_directory, jsonify, session, current_app
from app.main import bp
from app.models.models import Project, ProjectQuestion, Document, EvaluationRun, EvaluationResponse, EvaluationStatus, APILog, ChatSession, ChatMessage, DocumentType  # Import ChatSession and ChatMessage
from app import db  # Import the db instance
import os
from anthropic import Anthropic
from config import Config
from threading import Thread
import pdfplumber
import time
from datetime import datetime, timedelta  # Add timedelta to the import
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from werkzeug.utils import secure_filename
from docx import Document as DocxDocument
from io import BytesIO

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('main/index.html', title=Config.APP_NAME)

@bp.route('/projects')
def projects():
    # Query only active projects from the database
    active_projects = Project.query.filter_by(active=True).all()
    return render_template('main/projects.html', 
                         projects=active_projects, 
                         title='Projects - ' + Config.APP_NAME)

@bp.route('/questions')
def questions():
    # Query all project questions from the database
    all_questions = ProjectQuestion.query.all()
    return render_template('main/questions.html', questions=all_questions, title='Questions - ' + Config.APP_NAME)

@bp.route('/questions/edit/<int:question_id>', methods=['GET', 'POST'])
def edit_question(question_id):
    question = ProjectQuestion.query.get_or_404(question_id)
    
    if request.method == 'POST':
        try:
            question.question = request.form['question']
            question.name = request.form['name']
            question.prompt = request.form['prompt']
            question.order = int(request.form['order'])  # Convert to integer
            db.session.commit()
            flash('Project question updated successfully!', 'success')
            return redirect(url_for('main.questions'))
        except ValueError:
            flash('Invalid order value. Please enter a number.', 'danger')
            return redirect(request.url)
        except Exception as e:
            flash(f'Error updating question: {str(e)}', 'danger')
            return redirect(request.url)

    return render_template('main/edit_question.html', question=question)

@bp.route('/projects/<int:project_id>')
def project_details(project_id):
    project = Project.query.get_or_404(project_id)  # Fetch the project by ID
    return render_template('main/project_details.html', project=project)  # Render the project details template 

@bp.route('/projects/<int:project_id>/upload', methods=['GET', 'POST'])
def upload_document(project_id):
    project = Project.query.get_or_404(project_id)
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        document_type = request.form.get('document_type')
        
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            try:
                # Get file extension
                file_ext = file.filename.rsplit('.', 1)[1].lower()
                
                # Read file content and get size
                file_content = file.read()
                file_size = len(file_content)  # Get file size in bytes
                
                # Reset file pointer for content extraction
                file.seek(0)
                
                # Extract text based on file type
                if file_ext == 'pdf':
                    content = extract_text_from_pdf(file)
                elif file_ext == 'txt':
                    content = file_content.decode('utf-8')
                elif file_ext in ['doc', 'docx']:
                    content = extract_text_from_doc(file)
                
                # Create new document
                document = Document(
                    project_id=project.id,
                    filename=secure_filename(file.filename),
                    file_type=file_ext,
                    document_type=document_type,
                    file_size=file_size,  # Add file size
                    content=content,
                    content_preview=content[:1000] if content else None  # Add preview
                )
                
                db.session.add(document)
                db.session.commit()
                
                flash('Document uploaded successfully', 'success')
                return redirect(url_for('main.project_details', project_id=project.id))
                
            except Exception as e:
                flash(f'Error processing document: {str(e)}', 'error')
                return redirect(request.url)
                
        else:
            flash('Invalid file type. Supported formats: PDF, TXT, DOC, DOCX', 'error')
            return redirect(request.url)
    
    return render_template('main/upload_document.html', 
                         project=project,
                         document_types=DocumentType.choices())

@bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename) 

@bp.route('/projects/<int:project_id>/delete_document/<int:document_id>', methods=['POST'])
def delete_document(project_id, document_id):
    document = Document.query.get_or_404(document_id)  # Fetch the document by ID
    db.session.delete(document)  # Delete the document from the database
    db.session.commit()  # Commit the changes
    flash('Document deleted successfully!', 'success')
    return redirect(url_for('main.project_details', project_id=project_id))  # Redirect back to project details 

@bp.route('/api/projects/<int:project_id>/process_document/<int:document_id>', methods=['POST'])
def process_document(project_id, document_id):
    document = Document.query.get_or_404(document_id)
    file_path = os.path.join('uploads', document.filename)
    print(f"File path: {file_path}")

    try:
        if document.file_type == 'application/pdf':
            # Handle PDF files
            with pdfplumber.open(file_path) as pdf:
                content = ""
                for page in pdf.pages:
                    content += page.extract_text() + "\n"
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = file.read()

        print(f"File content length: {len(content)}")
        document.content = content
        document.content_preview = content[:100]
        db.session.commit()
        return jsonify({'message': 'Document processed successfully!'}), 200

    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/projects/<int:project_id>/start_assessment', methods=['GET', 'POST'])
def start_assessment(project_id):
    project = Project.query.get_or_404(project_id)
    if request.method == 'POST':
        selected_model = request.form.get('model_select', Config.CLAUDE_MODEL)
        selected_questions = request.form.getlist('selected_questions')
        selected_documents = request.form.getlist('selected_documents')  # Get selected documents
        temperature = float(request.form.get('temperature', 0.7))
        
        if not selected_questions:
            flash('Please select at least one question for the assessment.', 'danger')
            return redirect(url_for('main.start_assessment', project_id=project.id))
            
        if not selected_documents:
            flash('Please select at least one document for the assessment.', 'danger')
            return redirect(url_for('main.start_assessment', project_id=project.id))
        
        # Create new evaluation run
        evaluation = EvaluationRun(
            project_id=project.id,
            system_prompt=Config.DEFAULT_SYSTEM_PROMPT,
            status=EvaluationStatus.IN_PROGRESS,
            model_used=selected_model,
            total_questions=len(selected_questions)
        )
        
        # Add selected documents to the evaluation
        selected_doc_objects = Document.query.filter(Document.id.in_(selected_documents)).all()
        evaluation.documents.extend(selected_doc_objects)
        
        db.session.add(evaluation)
        db.session.commit()
        
        # Start the evaluation process in a background thread
        thread = Thread(target=run_evaluation, 
                       args=(project.id, evaluation.id, selected_questions, temperature, selected_documents))
        thread.daemon = True
        thread.start()
        
        return render_template('main/assessment_progress.html',
                             project=project, 
                             total_questions=len(selected_questions), 
                             evaluation_id=evaluation.id)
    
    # GET request - show the form
    questions = ProjectQuestion.query.order_by(ProjectQuestion.order).all()
    return render_template('main/start_assessment.html', 
                         project=project, 
                         config=Config,
                         questions=questions)

@bp.route('/projects/<int:project_id>/evaluation/<int:evaluation_id>')
def view_assessment(project_id, evaluation_id):
    project = Project.query.get_or_404(project_id)
    evaluation = EvaluationRun.query.get_or_404(evaluation_id)
    return render_template('main/view_assessment.html', project=project, evaluation=evaluation)

@bp.route('/setup/add_test_questions')
def add_test_questions():
    # Check if questions already exist
    if ProjectQuestion.query.first() is not None:
        flash('Test questions already exist!', 'info')
        return redirect(url_for('main.questions'))
    
    # Add test questions
    test_questions = [
        {
            'name': 'Overall Impact',
            'question': 'What was the overall impact of the project?',
            'prompt': 'Based on the provided documentation, analyze and evaluate the overall impact of this project. Consider both intended and unintended outcomes.',
            'order': 1
        },
        {
            'name': 'Effectiveness',
            'question': 'How effective was the project in achieving its objectives?',
            'prompt': 'Evaluate the effectiveness of the project in achieving its stated objectives. Provide specific examples from the documentation.',
            'order': 2
        },
        {
            'name': 'Sustainability',
            'question': 'How sustainable are the project outcomes?',
            'prompt': 'Assess the sustainability of the project outcomes. Consider factors such as institutional capacity, financial sustainability, and long-term viability.',
            'order': 3
        },
        {
            'name': 'Lessons Learned',
            'question': 'What are the key lessons learned from this project?',
            'prompt': 'Identify and analyze the key lessons learned from this project. What worked well, what could have been done differently, and what recommendations can be made for similar projects?',
            'order': 4
        }
    ]
    
    for q in test_questions:
        question = ProjectQuestion(**q)
        db.session.add(question)
    
    db.session.commit()
    flash('Test questions added successfully!', 'success')
    return redirect(url_for('main.questions')) 

@bp.route('/projects/assessment_progress/<int:evaluation_id>')
def check_assessment_progress(evaluation_id):
    evaluation = EvaluationRun.query.get_or_404(evaluation_id)
    responses = EvaluationResponse.query.filter_by(evaluation_run_id=evaluation_id).all()
    questions_answered = len(responses)
    
    # Get token usage for this evaluation
    token_stats = db.session.query(
        db.func.sum(APILog.input_tokens).label('total_input'),
        db.func.sum(APILog.output_tokens).label('total_output')
    ).filter(
        APILog.evaluation_id == evaluation_id,
        APILog.success == True
    ).first()
    
    # Calculate average time per question for estimation
    if questions_answered > 0:
        avg_time = db.session.query(
            db.func.avg(APILog.end_time - APILog.start_time)
        ).filter(
            APILog.evaluation_id == evaluation_id,
            APILog.success == True
        ).scalar()
        estimated_remaining = (evaluation.total_questions - questions_answered) * (avg_time.total_seconds() if avg_time else 0)
    else:
        estimated_remaining = 0
    
    return jsonify({
        'status': evaluation.status.value,
        'questions_answered': questions_answered,
        'total_questions': evaluation.total_questions,
        'is_complete': evaluation.status == EvaluationStatus.COMPLETED,
        'model_used': evaluation.model_used,
        'status_message': evaluation.status_message,
        'token_usage': {
            'input': token_stats.total_input or 0,
            'output': token_stats.total_output or 0
        },
        'estimated_remaining': round(estimated_remaining)
    })

def run_evaluation(project_id, evaluation_id, selected_question_ids, temperature, selected_documents):
    """Run the evaluation process in the background"""
    from app import create_app
    app = create_app()
    
    with app.app_context():
        try:
            evaluation = EvaluationRun.query.get(evaluation_id)
            project = Project.query.get(project_id)
            
            # Set rate limit based on model
            RATE_LIMITS = {
                'claude-3-opus-latest': 40000,
                'claude-3-sonnet-latest': 80000,
                'claude-3-haiku-latest': 100000
            }
            rate_limit = RATE_LIMITS.get(evaluation.model_used, 100000)
            buffer_limit = rate_limit - 5000  # Leave 5k token safety margin
            
            # Get selected documents with their types
            documents = Document.query.filter(Document.id.in_(selected_documents)).all()
            
            # Build context from documents
            document_contexts = []
            for doc in documents:
                doc_type = doc.document_type.replace('_', ' ').title()
                doc_context = f"\n=== {doc_type}: {doc.filename} ===\n{doc.content}\n"
                document_contexts.append(doc_context)
            
            # Combine all document contexts
            combined_context = "\n".join(document_contexts)
            
            # Initialize Anthropic client
            client = Anthropic(api_key=Config.CLAUDE_API_KEY)
            
            # Get questions in order
            questions = ProjectQuestion.query\
                .filter(ProjectQuestion.id.in_(selected_question_ids))\
                .order_by(ProjectQuestion.order)\
                .all()
            
            for question in questions:
                # Check rate limit before proceeding
                recent_tokens = get_recent_tokens()
                if recent_tokens > buffer_limit:
                    wait_message = f"Rate limit approaching ({recent_tokens:,}/{rate_limit:,}). Pausing for 60 seconds..."
                    print(wait_message)
                    evaluation.status_message = wait_message
                    db.session.commit()
                    time.sleep(60)
                    evaluation.status_message = None
                    db.session.commit()
                
                try:
                    # Create API log entry
                    api_log = APILog(
                        project_id=project.id,
                        evaluation_id=evaluation.id,
                        question_id=question.id,
                        model_used=evaluation.model_used,
                        project_name=project.name,
                        question_name=question.name,
                        start_time=datetime.utcnow()
                    )
                    db.session.add(api_log)
                    db.session.commit()
                    
                    # Build the prompt with document context
                    system_prompt = (
                        f"{Config.DEFAULT_SYSTEM_PROMPT}\n\n"
                        f"Project Context:\n"
                        f"Project Name: {project.name}\n"
                        f"Project Objectives: {project.key_project_objectives}\n\n"
                        f"Available Documents:\n{combined_context}\n\n"
                        f"Question: {question.question}\n"
                        f"Detailed Instructions: {question.prompt}"
                    )
                    
                    # Print the complete prompt
                    print("\n=== COMPLETE PROMPT ===")
                    print(system_prompt)
                    print("=== END PROMPT ===\n")
                    
                    # Call Claude API
                    response = client.messages.create(
                        model=evaluation.model_used,
                        temperature=temperature,
                        max_tokens=4000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": "Please provide your analysis based on the available documents."}]
                    )
                    
                    # Update API log with token usage
                    total_tokens = response.usage.input_tokens + response.usage.output_tokens
                    api_log.input_tokens = response.usage.input_tokens
                    api_log.output_tokens = response.usage.output_tokens
                    api_log.success = True
                    api_log.end_time = datetime.utcnow()
                    
                    # Create evaluation response
                    evaluation_response = EvaluationResponse(
                        evaluation_run_id=evaluation.id,
                        question_id=question.id,
                        response_text=response.content[0].text
                    )
                    db.session.add(evaluation_response)
                    db.session.commit()
                    
                    # Adaptive delay based on token usage
                    if total_tokens > 30000:
                        time.sleep(5)
                    else:
                        time.sleep(2)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error processing question {question.id}: {error_msg}")
                    
                    api_log.success = False
                    api_log.error_message = error_msg
                    api_log.end_time = datetime.utcnow()
                    db.session.commit()
                    
                    if "rate_limit_error" in error_msg.lower():
                        print("Rate limit hit, waiting 60 seconds...")
                        time.sleep(60)
                        continue
                    else:
                        evaluation.status = EvaluationStatus.FAILED
                        db.session.commit()
                        raise e
            
            evaluation.status = EvaluationStatus.COMPLETED
            db.session.commit()
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            evaluation = EvaluationRun.query.get(evaluation_id)
            if evaluation:
                evaluation.status = EvaluationStatus.FAILED
                db.session.commit()

def get_recent_tokens():
    """Get total tokens used in the last minute"""
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    return db.session.query(
        db.func.sum(APILog.input_tokens)
    ).filter(
        APILog.start_time >= one_minute_ago,
        APILog.success == True
    ).scalar() or 0

@bp.route('/projects/<int:project_id>/delete_assessment/<int:evaluation_id>', methods=['POST'])
def delete_assessment(project_id, evaluation_id):
    evaluation = EvaluationRun.query.get_or_404(evaluation_id)
    
    # Only delete evaluation responses
    EvaluationResponse.query.filter_by(evaluation_run_id=evaluation.id).delete()
    
    # Delete the evaluation itself
    db.session.delete(evaluation)
    db.session.commit()
    
    flash('Assessment deleted successfully!', 'success')
    return redirect(url_for('main.project_details', project_id=project_id)) 

@bp.route('/projects/toggle_status/<int:project_id>', methods=['POST'])
def toggle_project_status(project_id):
    project = Project.query.get_or_404(project_id)
    project.active = not project.active
    db.session.commit()
    status = "activated" if project.active else "deactivated"
    flash(f'Project {status} successfully!', 'success')
    return redirect(url_for('main.projects')) 

@bp.route('/admin/projects')
def admin_projects():
    # Query all projects, including inactive ones
    all_projects = Project.query.order_by(Project.name).all()
    return render_template('main/admin_projects.html', 
                         projects=all_projects, 
                         title='Manage Projects - ' + Config.APP_NAME) 

@bp.route('/admin/api-logs')
def api_logs():
    # Get summary statistics
    total_calls = APILog.query.count()
    successful_calls = APILog.query.filter_by(success=True).count()
    success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Get average token usage for successful calls
    token_stats = db.session.query(
        db.func.avg(APILog.input_tokens).label('avg_input'),
        db.func.avg(APILog.output_tokens).label('avg_output')
    ).filter(APILog.success == True).first()
    
    # Get average response time by model
    model_stats = db.session.query(
        APILog.model_used,
        db.func.count(APILog.id).label('calls'),
        db.func.avg(APILog.end_time - APILog.start_time).label('avg_time'),
        db.func.avg(APILog.input_tokens).label('avg_input'),
        db.func.avg(APILog.output_tokens).label('avg_output'),
        db.func.sum(db.case((APILog.success == True, 1), else_=0)).label('successes')
    ).filter(
        APILog.end_time.isnot(None)
    ).group_by(
        APILog.model_used
    ).all()
    
    # Format model stats for display
    model_performance = [{
        'model': model.replace('claude-', 'Claude ').replace('-latest', '').replace('-', '.'),
        'calls': calls,
        'avg_time': round(avg_time.total_seconds(), 2) if avg_time else 0,
        'avg_input': int(avg_input or 0),
        'avg_output': int(avg_output or 0),
        'success_rate': (successes / calls * 100) if calls > 0 else 0
    } for model, calls, avg_time, avg_input, avg_output, successes in model_stats]
    
    # Get overall average response time
    avg_response_time = db.session.query(
        db.func.avg(APILog.end_time - APILog.start_time)
    ).filter(APILog.success == True).scalar() or timedelta(seconds=0)
    
    # Get detailed logs with related data
    logs = db.session.query(APILog, Project, ProjectQuestion)\
        .join(Project, APILog.project_id == Project.id)\
        .join(ProjectQuestion, APILog.question_id == ProjectQuestion.id)\
        .order_by(APILog.start_time.desc())\
        .all()
    
    return render_template('main/api_logs.html',
                         logs=logs,
                         total_calls=total_calls,
                         success_rate=success_rate,
                         avg_input_tokens=token_stats.avg_input,
                         avg_output_tokens=token_stats.avg_output,
                         avg_response_time=avg_response_time,
                         model_performance=model_performance,
                         title='API Logs - ' + Config.APP_NAME) 

@bp.route('/ai-agent', methods=['GET', 'POST'])
def ai_agent():
    # Get or create chat session
    session_id = session.get('chat_session_id')
    if session_id:
        chat_session = ChatSession.query.get(session_id)
    else:
        chat_session = ChatSession()
        db.session.add(chat_session)
        db.session.commit()
        session['chat_session_id'] = chat_session.id
    
    if request.method == 'POST':
        user_query = request.form.get('query')
        
        # Create user message
        user_message = ChatMessage(
            session_id=chat_session.id,
            role='user',
            content=user_query
        )
        db.session.add(user_message)
        
        # Create SQLDatabase instance
        db_uri = current_app.config['SQLALCHEMY_DATABASE_URI']
        sql_db = SQLDatabase.from_uri(db_uri)
        
        # Initialize the LLM
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=Config.CLAUDE_API_KEY,
            temperature=0.2,
            max_tokens=4000
        )
        
        try:
            agent = create_sql_agent(
                llm=llm,
                toolkit=SQLDatabaseToolkit(db=sql_db, llm=llm),
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # Execute the agent and capture the response
            response = agent.run(user_query)
            
            # Create assistant message
            assistant_message = ChatMessage(
                session_id=chat_session.id,
                role='assistant',
                content=response
            )
            db.session.add(assistant_message)
            
            # Update session title if it's the first message
            if not chat_session.title:
                chat_session.title = user_query[:50] + "..."
            
            db.session.commit()
            
            # Get all messages for this session
            messages = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at).all()
            
            return render_template('main/ai_agent.html',
                                response=response,
                                chat_history=messages,
                                current_session=chat_session,
                                all_sessions=ChatSession.query.order_by(ChatSession.created_at.desc()).all())
                                
        except Exception as e:
            db.session.rollback()
            flash(f'Error processing query: {str(e)}', 'error')
            return redirect(url_for('main.ai_agent'))
    
    # Get all messages for this session
    messages = ChatMessage.query.filter_by(session_id=chat_session.id).order_by(ChatMessage.created_at).all()
    
    return render_template('main/ai_agent.html',
                         response=None,
                         chat_history=messages,
                         current_session=chat_session,
                         all_sessions=ChatSession.query.order_by(ChatSession.created_at.desc()).all())

@bp.route('/ai-agent/new-session')
def new_chat_session():
    session.pop('chat_session_id', None)
    return redirect(url_for('main.ai_agent'))

@bp.route('/ai-agent/session/<int:session_id>')
def load_chat_session(session_id):
    chat_session = ChatSession.query.get_or_404(session_id)
    session['chat_session_id'] = chat_session.id
    return redirect(url_for('main.ai_agent'))

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

def extract_text_from_pdf(file):
    """Extract text from a PDF file"""
    try:
        with pdfplumber.open(file) as pdf:
            text = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                    
                # Also extract text from tables
                for table in page.extract_tables():
                    table_text = []
                    for row in table:
                        row_text = [cell.strip() if cell else '' for cell in row]
                        if any(row_text):  # Only add non-empty rows
                            table_text.append(" | ".join(filter(None, row_text)))
                    if table_text:
                        text.append("\n".join(table_text))
            
            return "\n\n".join(text)
    except Exception as e:
        raise Exception(f"Error processing PDF file: {str(e)}")

def extract_text_from_doc(file):
    """Extract text from a DOC/DOCX file"""
    # Save the uploaded file object to a BytesIO object
    doc_bytes = BytesIO(file.read())
    
    try:
        # Open the document using python-docx
        doc = DocxDocument(doc_bytes)
        
        # Extract text from paragraphs
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Only add non-empty paragraphs
                text.append(paragraph.text)
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():  # Only add non-empty cells
                        row_text.append(cell.text.strip())
                if row_text:  # Only add non-empty rows
                    text.append(" | ".join(row_text))
        
        return "\n".join(text)
        
    except Exception as e:
        raise Exception(f"Error processing DOC/DOCX file: {str(e)}") 