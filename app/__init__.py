from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config
from flask_migrate import Migrate

# Initialize extensions
db = SQLAlchemy()
login = LoginManager()
login.login_view = 'auth.login'
migrate = Migrate()  # Initialize Flask-Migrate

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

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

    # Create database tables
    with app.app_context():
        db.create_all()

    @app.context_processor
    def inject_config():
        return dict(config=Config)

    return app

# Import models after db is defined
from app.models import models

# User loader function
@login.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))  # Assuming User is your user model 