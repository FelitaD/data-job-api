import flask
from flask import render_template
from flask_restful import Api
from datetime import datetime

from create_app import create_app
from resources.status import Status
from resources.user import UserRegister, UserLogin
from resources.job import Job, JobList
from resources.recommender import Recommender
from db.create_user_db import create_user_db

app = create_app()
api = Api(app)
api.add_resource(Status, '/status')
# api.add_resource(UserRegister, '/register')
api.add_resource(UserLogin, '/login')
api.add_resource(JobList, '/jobs')
api.add_resource(Job, '/job/<int:id>')
api.add_resource(Recommender, '/recommender')


@app.before_first_request
def register_users():
    create_user_db()


@app.route("/")
def hello():
    return render_template('index.html', now=datetime.now())


@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/jobs/')
def jobs():
    jobs = JobList().get().get('jobs')
    return render_template('jobs.html', jobs=jobs)


@app.route('/recommend/')
def recommend():
    recommendations = Recommender().get()[0].get('recommendations')
    print(recommendations)
    print(type(recommendations))
    return render_template('recommend.html', recommendations=recommendations)


if __name__ == '__main__':
    create_user_db()
    app.run(host='0.0.0.0', port=8000, debug=True)
