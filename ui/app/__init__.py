import os
import numpy as np
from flask import Flask
from config import Config
from flask_bootstrap import Bootstrap
from flask_session import Session

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)
Session(app)

from app.lenasampler import bp as lenasampler_bp
from app.eyegazecleaner import bp as eyegazecleaner_bp
app.register_blueprint(lenasampler_bp, url_prefix='/lenasampler')
app.register_blueprint(eyegazecleaner_bp, url_prefix='/eyegazecleaner')

from app import routes, models
