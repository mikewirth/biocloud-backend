import os
basedir = os.path.abspath(os.path.dirname(__file__))

from flask import Flask,render_template

app = Flask(__name__)
from app import views
