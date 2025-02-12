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
from langchain.agents import create_sql_agent, AgentExecutor
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
from flask_login import login_required, current_user
from app.auth.decorators import admin_required
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from sqlalchemy import desc, func, text, distinct, case, and_
from openai import OpenAI

@bp.route('/')
@bp.route('/index')
@login_required
def index():
    # Get all active projects
    projects = Project.query.filter_by(active=True).all()
    
    # Count projects with all required documents
    projects_with_docs = sum(
        1 for p in projects 
        if any(d.document_type == 'final_project_report' for d in p.documents) 
        and any(d.document_type == 'external_evaluation' for d in p.documents)
    )
    
    # Get total number of questions
    total_questions = ProjectQuestion.query.count()
    
    # Count projects with all questions answered
    projects_with_responses = sum(
        1 for p in projects
        if any(len(run.responses) == total_questions for run in p.evaluation_runs)
    )
    
    # Count projects with the most recent responses reviewed
    projects_reviewed = 0
    for p in projects:
        for run in p.evaluation_runs:
            # Create a dictionary to track the most recent response for each question
            latest_responses = {}
            for response in run.responses:
                question_id = response.question.id
                # Update the latest response if it's the most recent one
                if question_id not in latest_responses or response.created_at > latest_responses[question_id].created_at:
                    latest_responses[question_id] = response
            
            # Check if all latest responses are reviewed
            if all(latest_response.reviewed for latest_response in latest_responses.values()):
                projects_reviewed += 1
                break  # No need to check further runs for this project

    return render_template('main/index.html',
                         title=Config.APP_NAME,
                         projects=projects,
                         projects_with_docs=projects_with_docs,
                         projects_with_responses=projects_with_responses,
                         projects_reviewed=projects_reviewed)

@bp.route('/projects')
@login_required
def projects():
    search_query = request.args.get('search', '')
    sort_by = request.args.get('sort', 'default')

    # Get total number of questions
    total_questions = ProjectQuestion.query.count()

    # Subquery to get the most recent response for each question per project
    latest_responses = db.session.query(
        EvaluationResponse.project_id,
        EvaluationResponse.question_id,
        func.max(EvaluationResponse.created_at).label('max_created_at')
    ).group_by(
        EvaluationResponse.project_id,
        EvaluationResponse.question_id
    ).subquery()

    # Query to get response counts and review counts for each project
    response_stats = db.session.query(
        EvaluationResponse.project_id,
        func.count(distinct(EvaluationResponse.question_id)).label('answered_questions'),
        func.sum(case((EvaluationResponse.reviewed == True, 1), else_=0)).label('reviewed_count')
    ).join(
        latest_responses,
        and_(
            EvaluationResponse.project_id == latest_responses.c.project_id,
            EvaluationResponse.question_id == latest_responses.c.question_id,
            EvaluationResponse.created_at == latest_responses.c.max_created_at
        )
    ).group_by(
        EvaluationResponse.project_id
    ).subquery()

    # Query only active projects from the database
    active_projects = Project.query.filter_by(active=True).outerjoin(
        response_stats,
        Project.id == response_stats.c.project_id
    )

    # Apply search filter
    if search_query:
        active_projects = active_projects.filter(Project.integrity_partner_name.ilike(f'%{search_query}%'))

    # Apply sorting
    if sort_by == 'default' or sort_by == 'funding_round':
        active_projects = active_projects.order_by(Project.name_of_round.asc(), Project.file_number_db.asc())
    elif sort_by == 'file_number':
        active_projects = active_projects.order_by(Project.file_number_db.asc())
    elif sort_by == 'partner_name':
        active_projects = active_projects.order_by(Project.integrity_partner_name.asc())
    elif sort_by == 'region':
        active_projects = active_projects.order_by(Project.region.asc())
    elif sort_by == 'countries':
        active_projects = active_projects.order_by(Project.countries_covered.asc())

    active_projects = active_projects.all()

    # Convert query results to dictionary with tuple values
    stats_dict = {}
    for project_id, answered, reviewed in db.session.query(
        response_stats.c.project_id,
        response_stats.c.answered_questions,
        response_stats.c.reviewed_count
    ).all():
        stats_dict[project_id] = (answered or 0, reviewed or 0)

    return render_template('main/projects.html', 
                         projects=active_projects,
                         total_questions=total_questions,
                         response_counts=stats_dict,
                         title='Projects - ' + Config.APP_NAME)

@bp.route('/questions')
@login_required
def questions():
    # Query all project questions from the database
    all_questions = ProjectQuestion.query.all()
    return render_template('main/questions.html', questions=all_questions, title='Questions - ' + Config.APP_NAME)

@bp.route('/questions/edit/<int:question_id>', methods=['GET', 'POST'])
@login_required
@admin_required
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
@login_required
def project_details(project_id):
    project = Project.query.get_or_404(project_id)
    
    # Get the most recent response for each question using project_id
    latest_responses = db.session.query(
        EvaluationResponse
    ).filter(
        EvaluationResponse.project_id == project_id
    ).distinct(
        EvaluationResponse.question_id
    ).order_by(
        EvaluationResponse.question_id,
        EvaluationResponse.created_at.desc()
    ).all()

    return render_template('main/project_details.html', 
                         project=project,
                         latest_responses=latest_responses)

@bp.route('/projects/<int:project_id>/upload', methods=['GET', 'POST'])
@login_required
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
                
                if not content:
                    raise ValueError("No content extracted from file")

                print(f"Extracted content length: {len(content)}")  # Debug log
                
                # Create new document
                document = Document(
                    project_id=project.id,
                    filename=secure_filename(file.filename),
                    file_type=file_ext,
                    document_type=document_type,
                    file_size=file_size,
                    content_preview=content[:1000] if content else None
                )
                
                # Set encrypted content with error checking
                try:
                    document.set_content(content)
                    print(f"Content encrypted successfully")  # Debug log
                except Exception as e:
                    raise ValueError(f"Encryption failed: {str(e)}")
                
                db.session.add(document)
                db.session.commit()
                
                # Verify content was saved
                saved_doc = Document.query.get(document.id)
                if not saved_doc._content:
                    raise ValueError("Content was not saved to database")
                
                flash('Document uploaded successfully', 'success')
                return redirect(url_for('main.project_details', project_id=project.id))
                
            except Exception as e:
                db.session.rollback()
                flash(f'Error processing document: {str(e)}', 'error')
                print(f"Upload error: {str(e)}")  # Debug log
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

@bp.route('/projects/<int:project_id>/documents/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_document(project_id, document_id):
    try:
        from sqlalchemy import text
        
        # First check if document exists and belongs to this project
        document = Document.query.filter_by(id=document_id, project_id=project_id).first_or_404()
        
        # Remove any references in evaluation_documents
        db.session.execute(
            text('DELETE FROM univ_evaluation_documents WHERE document_id = :doc_id'),
            {'doc_id': document_id}
        )
        
        # Delete the document
        db.session.delete(document)
        db.session.commit()
        
        flash('Document deleted successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting document: {str(e)}', 'danger')
        print(f"Error deleting document: {str(e)}")
    
    return redirect(url_for('main.project_details', project_id=project_id))

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
@login_required
def start_assessment(project_id):
    project = Project.query.get_or_404(project_id)
    if request.method == 'POST':
        selected_model = request.form.get('model_select', Config.CLAUDE_MODEL)
        selected_questions = request.form.getlist('selected_questions')
        selected_documents = request.form.getlist('selected_documents')
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
@login_required
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
@login_required
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
            
            # Get selected documents with their types
            documents = Document.query.filter(Document.id.in_(selected_documents)).all()
            
            # Build context from documents with caching instruction
            document_contexts = []
            for doc in documents:
                doc_type = doc.document_type.replace('_', ' ').title()
                # Use get_content() instead of accessing content directly
                content = doc.get_content()
                if content:  # Only add if content was successfully decrypted
                    doc_context = f"\n=== {doc_type}: {doc.filename} ===\n{content}\n"
                    document_contexts.append(doc_context)
                else:
                    print(f"Warning: Could not decrypt content for document {doc.id}: {doc.filename}")
            
            if not document_contexts:
                raise ValueError("No readable document content found")

            # Combine all document contexts with instruction
            file_instruction = ("The following content is from project documents. "
                              "Use this information when responding to questions. "
                              "Reference specific documents when citing information:\n\n")
            combined_context = file_instruction + "\n".join(document_contexts)
            
            # Initialize Anthropic client with caching headers
            client = Anthropic(
                api_key=Config.CLAUDE_API_KEY,
                default_headers={
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31"
                }
            )
            
            # Create the system content with caching
            system = [
                {
                    "type": "text",
                    "text": Config.DEFAULT_SYSTEM_PROMPT
                },
                {
                    "type": "text",
                    "text": combined_context,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
            
            # Process each question
            for question in ProjectQuestion.query.filter(ProjectQuestion.id.in_(selected_question_ids)).order_by(ProjectQuestion.order):
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
                    
                    # Call Claude API
                    response = client.messages.create(
                        model=evaluation.model_used,
                        temperature=temperature,
                        max_tokens=4000,
                        system=system,
                        messages=[{
                            "role": "user",
                            "content": f"""
Project Context:
Project Name: {project.name}
Project Objectives: {project.key_project_objectives}

Question: {question.question}
Detailed Instructions: {question.prompt}

Please provide your analysis based on the available documents."""
                        }]
                    )
                    
                    # Update API log with token usage and cost
                    total_tokens = response.usage.input_tokens + response.usage.output_tokens
                    api_log.input_tokens = response.usage.input_tokens
                    api_log.output_tokens = response.usage.output_tokens
                    api_log.cache_tokens = len(combined_context.split())  # Approximate token count for cached content
                    
                    # Calculate cost
                    model_costs = Config.CLAUDE_COSTS.get(evaluation.model_used, Config.CLAUDE_COSTS['claude-3-haiku-20240307'])
                    input_cost = (api_log.input_tokens / 1000) * model_costs['input']
                    output_cost = (api_log.output_tokens / 1000) * model_costs['output']
                    cache_cost = (api_log.cache_tokens / 1000) * model_costs['cache']
                    api_log.cost_usd = input_cost + output_cost + cache_cost
                    
                    api_log.success = True
                    api_log.end_time = datetime.utcnow()
                    
                    # Create evaluation response
                    evaluation_response = EvaluationResponse(
                        evaluation_run_id=evaluation.id,
                        question_id=question.id,
                        project_id=project_id,
                        response_text=response.content[0].text,
                        reviewed=False,
                        created_at=datetime.utcnow()
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
@login_required
@admin_required
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
@login_required
@admin_required
def toggle_project_status(project_id):
    project = Project.query.get_or_404(project_id)
    project.active = not project.active
    db.session.commit()
    status = "activated" if project.active else "deactivated"
    flash(f'Project {status} successfully!', 'success')
    return redirect(url_for('main.projects')) 

@bp.route('/admin/projects')
@login_required
@admin_required
def admin_projects():
    # Query all projects, including inactive ones
    all_projects = Project.query.order_by(Project.name).all()
    return render_template('main/admin_projects.html', 
                         projects=all_projects, 
                         title='Manage Projects - ' + Config.APP_NAME) 

@bp.route('/admin/api-logs')
@login_required
@admin_required
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
@login_required
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
        
        # Initialize the LLM with higher temperature
        llm = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            anthropic_api_key=Config.CLAUDE_API_KEY,
            temperature=0.3,  
            max_tokens=4000
        )
        
        try:
            # Create a SQL generation prompt
            sql_prompt = f"""Based on this question, generate a SQL query to get the answer from our database:
            Question: {user_query}
            
            Database Structure:
            1. univ_projects (p):
               this table contains the project details
               - id (primary key)
               - name, name_of_round, file_number_db, scope, region, countries_covered
               - integrity_partner_name
               - funding_amount_usd (in millions)
               - start_year, end_year
            
            2. univ_evaluation_runs (er):
                this table contains the details of assessment runs for each project
               - id (primary key)
               - project_id (foreign key to univ_projects.id)
               - status, created_at, model_used
            
            3. univ_evaluation_responses (r):
                this table contains the responses to each question for each assessment run
               - id (primary key)
               - evaluation_run_id (foreign key to univ_evaluation_runs.id)
               - question_id (foreign key to univ_project_questions.id)
               - response_text (contains the actual response)
            
            4. univ_project_questions (q):
                this table contains the questions and their details
                look at the question and name fields to determine which question answers the user query, then match it with the assessment run for the project(s) idnetified in the user query
               - id (primary key)
               - question, name, prompt, order
               - here are the questions ids and topics:
                id = 1   2.1 b Intended outputs achieved
                id = 2   2.1 a Outputs by Category
                id = 3   2.2 Attributable Outcomes
                id = 4   2.3 Impact Evidence Analysis
                id = 5   2.4 Unplanned Results
                id = 6   2.5 Sustainable Results
                id = 7   2.6 Internal Factors
                id = 8   2.7 External Factors
                id = 9   Project Overview
                id = 11  Contributions and Ratings
            
            
            Common joins:
            - Join projects to evaluation runs: p.id = er.project_id
            - Join evaluation runs to responses: er.id = r.evaluation_run_id
            - Join responses to questions: r.question_id = q.id
            
            Return only the SQL query, nothing else."""
            
            # Get SQL query
            sql_response = llm.invoke(sql_prompt)
            sql_query = sql_response.content.strip()
            
            # Execute query
            result = sql_db.run(sql_query)
            
            # Generate natural language response
            analysis_prompt = f"""Original question: {user_query}

            Query used:
            {sql_query}

            Here is the data from our database that answers the question:
            Result: {result}
            
            You are an expert analyst helping to understand project performance. Use the data to provide 
            insightful analysis that directly answers the original question. Focus on extracting meaningful 
            insights from the data rather than just summarizing it. If patterns or trends emerge across 
            multiple projects, highlight those.

            Format numbers nicely and use proper sentence structure.
            If the question is not answered in the data, say so.
            If the question is not relevant to the data, say so.
            If the question is not clear, say so.
            If the question is not answerable, say so.
            
            Present your response in two parts:
            1. Your analytical response to the question
            2. The SQL query that was used (for verification)
            """
            
            # Get final response
            final_response = llm.invoke(analysis_prompt)
            response = final_response.content
            
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
            messages = ChatMessage.query.filter_by(session_id=chat_session.id)\
                .order_by(ChatMessage.created_at.desc())\
                .all()
            
            return render_template('main/ai_agent.html',
                                response=response,
                                chat_history=messages,
                                current_session=chat_session,
                                all_sessions=ChatSession.query.order_by(ChatSession.created_at.desc()).all())
                                
        except Exception as e:
            db.session.rollback()
            flash(f'Error processing query: {str(e)}', 'error')
            return redirect(url_for('main.ai_agent'))
    
    # GET request handling remains the same
    messages = ChatMessage.query.filter_by(session_id=chat_session.id)\
        .order_by(ChatMessage.created_at.desc())\
        .all()
    
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

@bp.route('/project/<int:project_id>/update_notes', methods=['POST'])
@login_required
def update_notes(project_id):
    project = Project.query.get_or_404(project_id)
    
    # Get the notes from the form
    notes = request.form.get('notes', '').strip()
    
    # Update the project
    project.notes = notes
    db.session.commit()
    
    # Return success
    return jsonify({'success': True, 'message': 'Notes updated successfully'})

@bp.route('/project/<int:project_id>/update', methods=['POST'])
@login_required
def update_project(project_id):
    project = Project.query.get_or_404(project_id)
    
    try:
        # Update existing fields
        project.name_of_round = float(request.form.get('name_of_round'))
        project.file_number_db = request.form.get('file_number_db')
        project.scope = request.form.get('scope')
        project.region = request.form.get('region')
        project.countries_covered = request.form.get('countries_covered')
        project.integrity_partner_name = request.form.get('integrity_partner_name')
        project.partner_type = request.form.get('partner_type')
        project.project_partners = request.form.get('project_partners')
        project.wb_or_eib = request.form.get('wb_or_eib')
        
        # Add missing fields
        project.key_project_objectives = request.form.get('key_project_objectives')
        project.sectoral_scope = request.form.get('sectoral_scope')
        project.specific_sector = request.form.get('specific_sector')
        project.funding_amount_usd = float(request.form.get('funding_amount_usd')) if request.form.get('funding_amount_usd') else None
        project.duration = request.form.get('duration')
        project.start_year = int(request.form.get('start_year')) if request.form.get('start_year') else None
        project.end_year = int(request.form.get('end_year')) if request.form.get('end_year') else None
        project.wb_income_classification = request.form.get('wb_income_classification')
        project.corruption_quintile = request.form.get('corruption_quintile')
        project.cci = float(request.form.get('cci')) if request.form.get('cci') else None
        project.government_type_eiu = request.form.get('government_type_eiu')
        project.government_score_eiu = float(request.form.get('government_score_eiu')) if request.form.get('government_score_eiu') else None
        
        # Handle checkboxes
        project.final_external_evaluation = 'final_external_evaluation' in request.form
        project.final_report = 'final_report' in request.form
        project.full_proposal = 'full_proposal' in request.form
        project.workplan = 'workplan' in request.form
        project.baseline_assessment = 'baseline_assessment' in request.form
        
        # Handle optional other field
        project.other = request.form.get('other')

        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}) 

@bp.route('/test-prompt', methods=['GET', 'POST'])
@login_required
def test_prompt():
    response_text = None
    if request.method == 'POST':
        user_prompt = request.form.get('prompt', '')
        if user_prompt:
            try:
                client = OpenAI(api_key=Config.OPENAI_API_KEY)
                full_prompt = Config.DEFAULT_SYSTEM_PROMPT + "\n\n" + user_prompt
                
                response = client.chat.completions.create(
                    model="o3-mini",
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                
                response_text = response.choices[0].message.content
                
            except Exception as e:
                flash(f'Error: {str(e)}', 'error')
    
    return render_template('main/test_prompt.html', 
                         response=response_text,
                         title='Test Prompt') 

@bp.route('/mark_response_reviewed/<int:response_id>', methods=['POST'])
@login_required
def mark_response_reviewed(response_id):
    response = EvaluationResponse.query.get_or_404(response_id)
    
    try:
        response.reviewed = True
        response.reviewed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500 

@bp.route('/update_response/<int:response_id>', methods=['POST'])
@login_required
def update_response(response_id):
    response = EvaluationResponse.query.get_or_404(response_id)
    
    try:
        data = request.get_json()
        response.response_text = data.get('response_text')
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500 
