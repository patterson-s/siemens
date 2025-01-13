from flask import render_template, redirect, url_for, request, flash, send_from_directory, jsonify
from app.main import bp
from app.models.models import Project, ProjectQuestion, Document, EvaluationRun, EvaluationResponse, EvaluationStatus, APILog  # Import EvaluationStatus
from app import db  # Import the db instance
import os
from anthropic import Anthropic
from config import Config
from threading import Thread
import pdfplumber
import time
from datetime import datetime, timedelta  # Add timedelta to the import

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
    question = ProjectQuestion.query.get_or_404(question_id)  # Fetch the question by ID
    if request.method == 'POST':
        question.question = request.form['question']  # Update the question field
        question.name = request.form['name']  # Update the name field
        question.prompt = request.form['prompt']  # Update the prompt field
        question.order = request.form['order']  # Update the order field
        db.session.commit()  # Save changes to the database
        flash('Project question updated successfully!', 'success')
        return redirect(url_for('main.questions'))  # Redirect to the questions list

    return render_template('main/edit_question.html', question=question)  # Render the edit form 

@bp.route('/projects/<int:project_id>')
def project_details(project_id):
    project = Project.query.get_or_404(project_id)  # Fetch the project by ID
    return render_template('main/project_details.html', project=project)  # Render the project details template 

@bp.route('/projects/<int:project_id>/upload', methods=['GET', 'POST'])
def upload_document(project_id):
    project = Project.query.get_or_404(project_id)  # Fetch the project by ID
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file:
            # Check for duplicate documents
            existing_document = Document.query.filter_by(
                project_id=project.id,
                filename=file.filename
            ).first()

            if existing_document:
                flash('A document with this name already exists for this project.', 'danger')
                return redirect(request.url)

            # Save the file to the server
            filename = file.filename
            file_path = os.path.join('uploads', filename)  # Define your upload folder
            file.save(file_path)

            # Create a new Document entry
            new_document = Document(
                project_id=project.id,
                filename=filename,
                file_type=file.content_type,
                file_size=os.path.getsize(file_path)
            )
            db.session.add(new_document)
            db.session.commit()

            # Process the document to extract content
            process_document(project.id, new_document.id)  # Call the processing function

            flash('Document uploaded successfully!', 'success')
            return redirect(url_for('main.project_details', project_id=project.id))

    return render_template('main/upload_document.html', project=project)  # Render the upload form 

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
        
        if not selected_questions:
            flash('Please select at least one question for the assessment.', 'danger')
            return redirect(url_for('main.start_assessment', project_id=project.id))
        
        # Create new evaluation run
        evaluation = EvaluationRun(
            project_id=project.id,
            system_prompt=Config.DEFAULT_SYSTEM_PROMPT,
            status=EvaluationStatus.IN_PROGRESS,
            model_used=selected_model,
            total_questions=len(selected_questions)  # Store the total number of questions
        )
        db.session.add(evaluation)
        db.session.commit()
        
        # Start the evaluation process in a background thread
        thread = Thread(target=run_evaluation, args=(project.id, evaluation.id, selected_questions))
        thread.daemon = True
        thread.start()
        
        total_questions = len(selected_questions)
        return render_template('main/assessment_progress.html',
                             project=project, 
                             total_questions=total_questions, 
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

def run_evaluation(project_id, evaluation_id, selected_question_ids):
    """Run the evaluation process in the background"""
    from app import create_app
    
    app = create_app()
    
    with app.app_context():
        try:
            evaluation = EvaluationRun.query.get(evaluation_id)
            project = Project.query.get(project_id)
            anthropic = Anthropic(api_key=Config.CLAUDE_API_KEY)
            
            # Track tokens used in this evaluation
            total_tokens = 0
            evaluation.status_message = None  # Add this field to store status messages
            db.session.commit()
            
            # Set rate limit based on model
            RATE_LIMITS = {
                'claude-3-opus-latest': 40000,
                'claude-3-sonnet-latest': 80000,
                'claude-3-haiku-latest': 100000
            }
            rate_limit = RATE_LIMITS.get(evaluation.model_used, 100000)  # Default to 100k if unknown
            buffer_limit = rate_limit - 5000  # Leave 5k token safety margin
            
            # Get tokens used in last minute across all evaluations
            def get_recent_tokens():
                one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                return db.session.query(
                    db.func.sum(APILog.input_tokens)
                ).filter(
                    APILog.start_time >= one_minute_ago,
                    APILog.success == True
                ).scalar() or 0
            
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
                    time.sleep(60)  # Wait for a minute
                    evaluation.status_message = None
                    db.session.commit()
                
                # Create API log entry
                api_log = APILog(
                    project_id=project_id,
                    evaluation_id=evaluation_id,
                    question_id=question.id,
                    model_used=evaluation.model_used,
                    start_time=datetime.utcnow(),
                    project_name=project.name,
                    question_name=question.name
                )
                db.session.add(api_log)
                db.session.commit()
                
                try:
                    documents_content = "\n\n".join([doc.content for doc in project.documents if doc.content])
                    
                    response = anthropic.messages.create(
                        model=evaluation.model_used,
                        max_tokens=4096,
                        system=evaluation.system_prompt,
                        messages=[
                            {
                                "role": "user",
                                "content": f"Document contents:\n{documents_content}\n\nQuestion: {question.prompt}"
                            }
                        ]
                    )
                    
                    # Update token tracking
                    total_tokens += response.usage.input_tokens
                    
                    # Update API log with success and token counts
                    api_log.input_tokens = response.usage.input_tokens
                    api_log.output_tokens = response.usage.output_tokens
                    api_log.success = True
                    api_log.end_time = datetime.utcnow()
                    
                    # Save the response
                    evaluation_response = EvaluationResponse(
                        evaluation_run_id=evaluation.id,
                        question_id=question.id,
                        response_text=response.content[0].text
                    )
                    db.session.add(evaluation_response)
                    db.session.commit()
                    
                    # Adaptive delay based on token usage
                    if total_tokens > 30000:
                        time.sleep(5)  # Longer delay when using lots of tokens
                    else:
                        time.sleep(2)  # Normal delay
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"Error processing question {question.id}: {error_msg}")
                    
                    # Update API log with error
                    api_log.success = False
                    api_log.error_message = error_msg
                    api_log.end_time = datetime.utcnow()
                    db.session.commit()
                    
                    if "rate_limit_error" in error_msg:
                        print("Rate limit hit, waiting 60 seconds before continuing...")
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