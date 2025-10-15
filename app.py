import os
import logging
from flask import Flask, render_template
#from dash import Dash, html, dcc
# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Home page route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Import predict Blueprint from routes
from routes.predict_route import predict_bp
app.register_blueprint(predict_bp)

from  services.dashbord import dashboard_bp
app.register_blueprint(dashboard_bp)
# MAIN
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
