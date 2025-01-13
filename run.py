from dotenv import load_dotenv, find_dotenv
import os

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True, encoding='utf-8')

from app import create_app

app = create_app()

if __name__ == '__main__':
    print("Starting the Flask application...")  # Debugging line
    app.run(debug=True) 