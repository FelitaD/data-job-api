from datetime import datetime


from db.db import db


class JobModel(db.Model):
    """
    Internal representation of the Job resource and helper.
    Client doesn't interact directly with these methods.
    """
    __tablename__ = 'processed_jobs'

    id = db.Column(db.Integer)
    url = db.Column(db.String(500), primary_key=True)
    title = db.Column(db.String(100))
    company = db.Column(db.String(100))
    stack = db.Column(db.String)
    remote = db.Column(db.String(100))
    location = db.Column(db.String(100))
    industry = db.Column(db.String(100))
    type = db.Column(db.String(100))
    created_at = db.Column(db.Date)
    text = db.Column(db.Text)
    summary = db.Column(db.Text)
    education = db.Column(db.String(100))
    experience = db.Column(db.String(100))
    size = db.Column(db.String(100))

    def __init__(self, id, url, title, company, stack, remote, location, industry, _type, created_at, text, summary, education, experience, size):
        self.id = id
        self.url = url
        self.title = title
        self.company = company
        self.stack = stack
        self.remote = remote
        self.location = location
        self.industry = industry
        self.type = _type
        self.created_at = created_at
        self.text = text
        self.summary = summary
        self.education = education
        self.experience = experience
        self.size = size

    def json(self):
        return {
                'id': self.id,
                'url': self.url,
                'title': self.title,
                'company': self.company,
                'stack': self.stack,
                'remote': self.remote,
                'location': self.location,
                'industry': self.industry,
                'type': self.type,
                'created_at': self.created_at.strftime("%m-%d-%Y"),
                'text': self.text,
                'summary': self.summary,
                'education': self.education,
                'experience': self.experience,
                'size': self.size,
                }

    @classmethod
    def find_by_id(cls, id):
        return JobModel.query.filter_by(id=id).first()  # returns JobModel instance with init attributes

    def save_to_db(self):
        """ Upsert : update or insert """
        db.session.add(self)
        db.session.commit()

    # def delete_from_db(self):
    #     db.session.delete(self)
    #     db.session.commit()
