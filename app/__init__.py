from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
from flask_migrate import Migrate
import os

# Initialize extensions
db = SQLAlchemy()
login = LoginManager()
login.login_view = 'auth.login'
migrate = Migrate()  # Initialize Flask-Migrate

def create_app(config_class=Config):
    app = Flask(__name__)
    
    # Debug prints before config
    print("Environment variables:")
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL')}")
    
    app.config.from_object(config_class)
    
    # Debug prints after config
    print("Flask config:")
    print(f"SQLALCHEMY_DATABASE_URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

    print("Database URL:", app.config['SQLALCHEMY_DATABASE_URI'])  # Debugging line

    # Add template filter for newlines
    @app.template_filter('nl2br')
    def nl2br(value):
        return value.replace('\n', '<br>\n') if value else ''

    # Initialize extensions with app
    db.init_app(app)
    login.init_app(app)
    migrate.init_app(app, db)  # Initialize Flask-Migrate with the app and db

    # Register blueprints
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)

    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    # Create database tables
    with app.app_context():
        db.create_all()

    @app.context_processor
    def inject_config():
        return dict(config=Config)

    return app

# Import models after db is defined
from app.models.models import User

# User loader function
@login.user_loader
def load_user(user_id):
    return User.query.get(int(user_id)) 