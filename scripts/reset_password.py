import os
import sys
from werkzeug.security import generate_password_hash

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python path:", sys.path)  # Debug print

from app import create_app, db
from app.models.models import User

def reset_user_password(username, password):
    print("Starting password reset process...")
    
    try:
        print("Creating app...")  # Debug print
        app = create_app()
        print("App created successfully")  # Debug print
        
        with app.app_context():
            print("Inside app context")  # Debug print
            
            # Find the user
            print(f"Looking for user: {username}")  # Debug print
            user = User.query.filter_by(username=username).first()
            if not user:
                print(f"User {username} not found!")
                return
            
            print(f"Found user: {user.username}")  # Debug print
            
            # Update password
            print("Setting new password...")  # Debug print
            user.set_password(password)
            db.session.commit()
            print("Password updated in database")  # Debug print
            
            # Verify the new password
            if user.check_password(password):
                print(f"Password successfully updated for user: {username}")
                print(f"New password hash: {user.password_hash}")
            else:
                print("Password verification failed!")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print full traceback
        raise e

if __name__ == '__main__':
    print("Script starting...")  # Debug print
    username = "univ"
    password = "inte6rity"
    
    print(f"Resetting password for user: {username}")
    reset_user_password(username, password)
    print("Script completed")  # Debug print 