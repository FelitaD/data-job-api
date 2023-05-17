from flask import Flask, Blueprint
from flask_jwt_extended import JWTManager

from definitions import DB_STRING
from db.db import db


def create_app():
    app = Flask(__name__)
    bp = Blueprint('greetings', __name__)
    app.register_blueprint(bp)

    app.config['PROPAGATE_EXCEPTIONS'] = True
    app.config['SECRET_KEY'] = b'_5#y@!L"F4gweg28z\n\xec]/'
    app.config['SQLALCHEMY_DATABASE_URI'] = DB_STRING

    app.config['TESTING'] = False

    with app.app_context():
        db.init_app(app)
        db.Model.metadata.reflect(db.engine)

    jwt = JWTManager(app)

    return app
