from flask_restful import Resource, reqparse
from flask_jwt_extended import jwt_required, get_jwt_identity

from models.recommender import RecommenderModel


class Recommender(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('job_id', type=int, required=True, help='Enter the id of a job you like.')
        self.parser.add_argument('columns', type=str, required=False, help='Enter the columns included in original dataframe.')
        self.parser.add_argument('feature_columns', type=str, required=False, help='Enter the feature_columns included in the 1st cosine similarity.')
        self.parser.add_argument('feature_names_weights', type=str, required=False, help='Enter the feature_names_weights included in 2nd the cosine similarity.')

        self.data = self.parser.parse_args()
        self.recommender = RecommenderModel(self.data['job_id'],
                                            self.data['columns'],
                                            self.data['feature_columns'],
                                            self.data['feature_names_weights'])

    def post(self):
        response = {'job_id': self.data['job_id'],
                    'similar_jobs': self.recommender.recommend().head(1).to_json()}
        return response, 200


    # def add_job_id(self):
        # df = recommender.compute_mean_similarities(original_df, job_ids)
        # # df = df.sort_values(by='mean_similarity', ascending=False)  # Will shuffle index
        # print(df)
        # print(df.iloc[333])
        # print(base_recommender.get_id_from_index(df, 333))