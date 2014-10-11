import os
basedir = os.path.abspath(os.path.dirname(__file__))

from flask import Flask,render_template
from flask.ext.cors import CORS

app = Flask(__name__)
# Set CORS options on app configuration
app.config['CORS_HEADERS'] = "Origin, X-Requested-With,Content-Type, Accept"
app.config['PROPAGATE_EXCEPTIONS']=True
app.config['CORS_RESOURCES'] = {r"/render": {"origins": "localhost:3000"}}

cors = CORS(app)

from app import views
