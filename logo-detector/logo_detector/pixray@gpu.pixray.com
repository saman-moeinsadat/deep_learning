import logging
import json
import time

from flask import Flask, request, render_template
from flask_cors import CORS

from logo_detector.config import Config
from logo_detector.detector import detect_resnet


app = Flask(__name__)
log = logging.getLogger()
log.setLevel(logging.INFO)


# App Factory
def create_app(config_class=Config):
    app = Flask(__name__)
    CORS(app)
    app.config.from_object(config_class)

    @app.route("/", methods=["GET"])
    def root():
        return """<form method="POST">
<input name="image_url">
<input type="submit">
</form>"""

    @app.route("/", methods=["POST"])
    def predict():
        since = time.time()
        image_url = request.form['image_url']
        logo = detect_resnet(image_url)
        return """<h1> This is "{}"! in "{}" ms</h1>
<h1><img src="{}"></h1>
""".format(logo, int((time.time() - since) * 1000), image_url)
    return app


APP = create_app()
