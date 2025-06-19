import enum
from datetime import datetime, timedelta  # Ensure datetime is imported for ProjectQuestion model
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash  # Add this for password hashing
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from cryptography.fernet import Fernet, InvalidToken
from flask import current_app
import logging
from config import Config
from base64 import b64encode
from sqlalchemy_utils import TSVectorType
from app import db

class User(UserMixin, db.Model):
    __tablename__ = 'univ_users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255))
    is_admin = db.Column(db.Boolean, nullable=False, default=False)
    failed_login_count = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_locked(self):
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False

    def increment_failed_login(self):
        self.failed_login_count += 1
        if self.failed_login_count >= 5:
            self.locked_until = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()

    def reset_failed_login(self):
        self.failed_login_count = 0
        self.locked_until = None
        db.session.commit()

    def __repr__(self):
        return f'<User {self.username}>'


class DocumentType(enum.Enum):
    external_evaluation = 'external_evaluation'
    final_project_report = 'final_project_report'
    original_proposal = 'original_proposal'

    @classmethod
    def choices(cls):
        return [(choice.value, choice.value.replace('_', ' ').title()) 
                for choice in cls]

class Document(db.Model):
    __tablename__ = 'univ_documents'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('univ_projects.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(100))
    document_type = db.Column(db.Enum(DocumentType), nullable=False, default=DocumentType.external_evaluation)
    file_size = db.Column(db.Integer)  # Word count
    content_preview = db.Column(db.Text)
    _content = db.Column('content', db.Text)  # Renamed to match EvaluationResponse pattern
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Remove the search_vector field
    # search_vector = db.Column(TSVectorType('content'))
    
    # Relationships
    project = db.relationship('Project')
    
    def get_content(self):
        """Decrypt and return content"""
        if not self._content:
            return None
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            return f.decrypt(self._content.encode()).decode()
        except Exception as e:
            current_app.logger.error(f"Decryption error: {str(e)}")
            return "Error: Could not decrypt document content"
    
    def set_content(self, content):
        """Encrypt and set content"""
        if not content:
            self._content = None
            return
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            self._content = f.encrypt(content.encode()).decode()
        except Exception as e:
            current_app.logger.error(f"Encryption error: {str(e)}")
            raise ValueError("Could not encrypt document content")

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
    name_of_round = db.Column(db.Numeric(3, 1), nullable=False)  # Changed to Numeric to handle values like 1.0, 2.5 etc
    name = db.Column(db.String(255), nullable=False)
    file_number_db = db.Column(db.Float)
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
    wb_income_classification = db.Column(db.String(50))
    corruption_quintile = db.Column(db.String(25))
    cci = db.Column(db.Numeric(4, 2))
    government_type_eiu = db.Column(db.String(100))
    government_score_eiu = db.Column(db.Numeric(4, 2))
    other = db.Column(db.Text)
    notes = db.Column(db.Text)
    active = db.Column(db.Boolean, default=True, nullable=False)
    
    # Numeric count fields
    num_pubpri_dialogues = db.Column(db.Numeric(5, 1))
    num_legal_contribuntions = db.Column(db.Numeric(5, 1))
    num_implement_mechanisms = db.Column(db.Numeric(5, 1))
    num_voluntary_standards = db.Column(db.Numeric(5, 1))
    num_voluntary_signatories = db.Column(db.Numeric(5, 1))
    num_organizations_supported = db.Column(db.Numeric(5, 1))
    
    # Rating fields
    rate_output_achieved = db.Column(db.Integer)
    rate_impact_evidence = db.Column(db.String(25))
    rate_sustainability = db.Column(db.Integer)
    rate_project_design = db.Column(db.Integer)
    rate_project_management = db.Column(db.String(50))
    rate_quality_evaluation = db.Column(db.String(25))
    rate_impact_progress = db.Column(db.Integer)
    rate_signif_frameworks = db.Column(db.String(50))
    rate_signif_practices = db.Column(db.String(50))
    rate_signif_capacity = db.Column(db.String(50))
    
    # Add these new fields
    num_new_courses = db.Column(db.Numeric(5, 1))
    num_individ_trained = db.Column(db.Numeric(5, 1))
    num_training_activities = db.Column(db.Numeric(5, 1))
    num_organizaed_events = db.Column(db.Numeric(5, 1))
    num_event_attendees = db.Column(db.Numeric(5, 1))
    num_publications = db.Column(db.Numeric(5, 1))
    
    # Update relationship definition to match Document model
    documents = db.relationship('Document', back_populates='project', lazy=True)
    evaluation_runs = db.relationship('EvaluationRun', backref='project', lazy=True)

class EvaluationStatus(enum.Enum):
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'

    @classmethod
    def choices(cls):
        return [(choice.value, choice.value.replace('_', ' ').title()) 
                for choice in cls]

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
    
    # Remove the backref from here
    responses = db.relationship('EvaluationResponse', lazy=True)
    
    # Keep other relationships
    documents = db.relationship('Document', 
                              secondary='univ_evaluation_documents',
                              lazy='dynamic')

# Add association table for many-to-many relationship
evaluation_documents = db.Table('univ_evaluation_documents',
    db.Column('evaluation_id', db.Integer, db.ForeignKey('univ_evaluation_runs.id'), primary_key=True),
    db.Column('document_id', db.Integer, db.ForeignKey('univ_documents.id'), primary_key=True)
)

class EvaluationResponse(db.Model):
    __tablename__ = 'univ_evaluation_responses'
    
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('univ_project_questions.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('univ_projects.id'), nullable=True)
    evaluation_run_id = db.Column(db.Integer, db.ForeignKey('univ_evaluation_runs.id'), nullable=False)
    _response_text = db.Column('response_text', db.Text, nullable=False)  # Rename to match Document model pattern
    reviewed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    question = db.relationship('ProjectQuestion', backref='responses')
    project = db.relationship('Project', backref='responses')
    evaluation_run = db.relationship('EvaluationRun')

    def get_response_text(self):
        """Decrypt and return response text"""
        if not self._response_text:
            return None
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            return f.decrypt(self._response_text.encode()).decode()
        except Exception as e:
            current_app.logger.error(f"Decryption error: {str(e)}")
            return "Error: Could not decrypt response"

    def set_response_text(self, value):
        """Encrypt and store response text"""
        if not value:
            self._response_text = None
            return
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            self._response_text = f.encrypt(value.encode()).decode()
        except Exception as e:
            current_app.logger.error(f"Encryption error: {str(e)}")
            raise ValueError(f"Could not encrypt response text: {str(e)}")

    # Property to maintain compatibility
    response_text = property(get_response_text, set_response_text)

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

class ChatSession(db.Model):
    __tablename__ = 'univ_chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    title = db.Column(db.String(255))  # Will be set to first few words of first message
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade='all, delete-orphan')

class ChatMessage(db.Model):
    __tablename__ = 'univ_chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('univ_chat_sessions.id'), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Optional: Link to related database queries/results
    sql_query = db.Column(db.Text)
    query_results = db.Column(db.Text)  # Store as JSON

class LoginLog(db.Model):
    __tablename__ = 'univ_login_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('univ_users.id', ondelete='SET NULL'), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    success = db.Column(db.Boolean, nullable=False)
    ip_address = db.Column(db.String(45))  # IPv6 can be up to 45 chars
    user_agent = db.Column(db.String(255))
    attempted_username = db.Column(db.String(64), nullable=True)
    
    user = db.relationship('User', backref='login_logs')

class Interview(db.Model):
    __tablename__ = 'univ_interviews'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    _content = db.Column('_content', db.Text)
    word_count = db.Column(db.Integer, nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey('univ_users.id'), nullable=False)
    uploaded_by = db.relationship('User', backref='interviews')

    @property
    def content(self):
        """Get decrypted content"""
        if not self._content:
            return None
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            return f.decrypt(self._content.encode()).decode('utf-8')
        except Exception as e:
            current_app.logger.error(f"Decryption error: {str(e)}")
            return None

    @content.setter 
    def content(self, value):
        """Set encrypted content"""
        if not value:
            self._content = None
            return
        try:
            f = Fernet(current_app.config['ENCRYPTION_KEY'].encode())
            self._content = f.encrypt(value.encode()).decode('utf-8')
        except Exception as e:
            current_app.logger.error(f"Encryption error: {str(e)}")
            raise ValueError(f"Could not encrypt content: {str(e)}")

    def __repr__(self):
        return f'<Interview {self.name}>'

class InterviewQuestion(db.Model):
    __tablename__ = 'univ_interview_questions'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    text = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<InterviewQuestion {self.title}>'

class DatabaseQuery(db.Model):
    __tablename__ = 'univ_database_queries'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    display_order = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    question_ids = db.Column(ARRAY(db.Integer), nullable=True)  # Array of question IDs

    def __repr__(self):
        return f'<DatabaseQuery {self.title}>'

DOCUMENT_TYPES = [
    ('original_proposal', 'Original Proposal'),
    ('final_project_report', 'Project Report'),
    ('external_evaluation', 'External Evaluation')
]