from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required, get_jwt_identity

from models.recommender import RecommenderModel


class Recommender(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('job_id', type=int, required=True, help='Enter the id of a job you like.')

        self.data = self.parser.parse_args()
        self.recommender = RecommenderModel(self.data['job_id'])

    def post(self):
        response = {'job_id': self.data['job_id'],
                    'similar_jobs': self.recommender.recommend(self.data['job_id'])}
        return response, 200


    # def add_job_id(self):
        # df = recommender.compute_mean_similarities(original_df, job_ids)
        # # df = df.sort_values(by='mean_similarity', ascending=False)  # Will shuffle index
        # print(df)
        # print(df.iloc[333])
        # print(base_recommender.get_id_from_index(df, 333))