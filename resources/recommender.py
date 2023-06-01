from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required, get_jwt_identity

from models.recommender import RecommenderModel


class Recommender(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('job_id', required=False, location='form',
                                 help='Enter one or multiple ids of jobs you like.')
        self.parser.add_argument('columns', type=str, required=False,
                                 help='Enter the columns included in original dataframe.')
        self.parser.add_argument('feature_columns', type=str, required=False,
                                 help='Enter the feature_columns included in the 1st cosine similarity.')
        self.parser.add_argument('feature_names_weights', type=str, required=False,
                                 help='Enter the feature_names_weights included in 2nd the cosine similarity.')
        self.data = self.parser.parse_args()
        self.recommender = RecommenderModel(self.data['job_id'],
                                            self.data['columns'],
                                            self.data['feature_columns'],
                                            self.data['feature_names_weights'])
        self.recommender.recommend()
        self.similar_jobs_json = self.recommender.format_json()

    def post(self):
        # TODO: to json
        response = {'similar_jobs': self.similar_jobs_json}
        return response, 200

    def get(self):
        # TODO: to html
        response = {'recommendations': self.recommender.original_df[['id', 'title', 'company', 'stack', 'remote', 'url', 'mean_similarity']].head(100).to_html()}
        return response, 200
