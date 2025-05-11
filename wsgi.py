# wsgi.py

import sys
import os

# Add the path to your Flask app to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'eye', 'eye', 'eye')))

# Import the Flask application from your app.py
from app import app as application

if __name__ == "__main__":
    application.run()
