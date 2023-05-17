import pandas as pd

from db.db import db


class UserModel(db.Model):

    __tablename__ = 'users'

    username = db.Column(db.String(20), primary_key=True)
    password = db.Column(db.String(40))
    v1 = db.Column(db.Boolean)
    v2 = db.Column(db.Boolean)

    def __init(self, username, password, v1, v2):
        self.username = username
        self.password = password
        self.v1 = v1
        self.v2 = v2

    @classmethod
    def find_by_username(cls, username):
        return cls.query.filter_by(username=username).first()
