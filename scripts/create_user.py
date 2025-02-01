import os
import sys
from werkzeug.security import generate_password_hash

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.models import User

def create_admin_user(username, password):
    print("Starting user creation process...")
    
    try:
        app = create_app()
        print("App created successfully")
        
        with app.app_context():
            print("Inside app context")
            
            # Check if user already exists
            user = User.query.filter_by(username=username).first()
            if user:
                print(f"User {username} already exists!")
                return
            
            print("Creating new user...")
            # Create new user
            user = User(
                username=username,
                is_admin=True
            )
            user.set_password(password)
            
            # Add and commit to database
            db.session.add(user)
            db.session.commit()
            
            print(f"Created admin user: {username}")
            print(f"Password hash: {user.password_hash}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise e

if __name__ == '__main__':
    print("Script started")
    
    if len(sys.argv) != 3:
        print("Usage: python create_user.py <username> <password>")
        sys.exit(1)
    
    username = sys.argv[1]
    password = sys.argv[2]
    
    print(f"Attempting to create user: {username}")
    create_admin_user(username, password) 