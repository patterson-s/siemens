from app import db  # Import the db instance
from datetime import datetime  # Ensure datetime is imported for ProjectQuestion model
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash  # Add this for password hashing
from enum import Enum

class User(UserMixin, db.Model):
    __tablename__ = 'univ_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    role = db.Column(db.String(20), nullable=False, default='user')
    is_appeal_writer = db.Column(db.Boolean, nullable=False, default=False)
    is_chat_user = db.Column(db.Boolean, nullable=False, default=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @property
    def is_admin(self):
        return self.role == 'admin'

class Document(db.Model):
    __tablename__ = 'univ_documents'

    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('univ_projects.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), nullable=False)  # e.g., 'pdf', 'txt', 'csv'
    file_size = db.Column(db.Integer, nullable=False)  # size in bytes
    content = db.Column(db.Text, nullable=True)  # For text-based files
    content_preview = db.Column(db.String(1000), nullable=True)  # Short preview of content
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)  # Date of upload

class ProjectQuestion(db.Model):
    __tablename__ = 'univ_project_questions'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    name = db.Column(db.String(50), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Project(db.Model):
    __tablename__ = 'univ_projects'
    
    id = db.Column(db.Integer, primary_key=True)
    name_of_round = db.Column(db.String(100), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    file_number_db = db.Column(db.Integer)
    scope = db.Column(db.String(100))
    region = db.Column(db.String(100))
    countries_covered = db.Column(db.String(255))
    integrity_partner_name = db.Column(db.String(255))
    partner_type = db.Column(db.String(50))
    project_partners = db.Column(db.Text)
    wb_or_eib = db.Column(db.String(50))
    key_project_objectives = db.Column(db.Text)
    sectoral_scope = db.Column(db.String(100))
    specific_sector = db.Column(db.String(100))
    funding_amount_usd = db.Column(db.Numeric(10, 2))
    duration = db.Column(db.String(50))
    start_year = db.Column(db.Integer)
    end_year = db.Column(db.Integer)
    final_external_evaluation = db.Column(db.Boolean)
    final_report = db.Column(db.Boolean)
    full_proposal = db.Column(db.Boolean)
    workplan = db.Column(db.Boolean)
    baseline_assessment = db.Column(db.Boolean)
    other = db.Column(db.Text)
    notes = db.Column(db.Text)
    description = db.Column(db.Text, nullable=True)
    system_instructions = db.Column(db.Text, nullable=True)
    active = db.Column(db.Boolean, default=True, nullable=False)
    documents = db.relationship('Document', backref='project', lazy=True)
    evaluation_runs = db.relationship('EvaluationRun', backref='project', lazy=True)

class EvaluationStatus(Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'

class EvaluationRun(db.Model):
    __tablename__ = 'univ_evaluation_runs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('univ_projects.id'), nullable=False)
    system_prompt = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Enum(EvaluationStatus), nullable=False, default=EvaluationStatus.PENDING)
    model_used = db.Column(db.String(50), nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    status_message = db.Column(db.String(255), nullable=True)
    
    responses = db.relationship('EvaluationResponse', backref='evaluation_run', lazy=True)

class EvaluationResponse(db.Model):
    __tablename__ = 'univ_evaluation_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    evaluation_run_id = db.Column(db.Integer, db.ForeignKey('univ_evaluation_runs.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('univ_project_questions.id'), nullable=False)
    response_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    question = db.relationship('ProjectQuestion')

class APILog(db.Model):
    __tablename__ = 'univ_api_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('univ_projects.id', ondelete='SET NULL'), nullable=True)
    evaluation_id = db.Column(db.Integer, db.ForeignKey('univ_evaluation_runs.id', ondelete='SET NULL'), nullable=True)
    question_id = db.Column(db.Integer, db.ForeignKey('univ_project_questions.id', ondelete='SET NULL'), nullable=True)
    model_used = db.Column(db.String(50), nullable=False)
    input_tokens = db.Column(db.Integer)
    output_tokens = db.Column(db.Integer)
    start_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    success = db.Column(db.Boolean, nullable=False, default=False)
    error_message = db.Column(db.Text)
    
    # Store names at time of logging in case referenced records are deleted
    project_name = db.Column(db.String(255))
    question_name = db.Column(db.String(50))
    
    # Optional relationships
    project = db.relationship('Project', backref='api_logs')
    evaluation = db.relationship('EvaluationRun', backref='api_logs')
    question = db.relationship('ProjectQuestion', backref='api_logs')