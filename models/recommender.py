import ast
import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine

USER = os.getenv('JOB_MARKET_DB_USER')
PWD = os.getenv('JOB_MARKET_DB_PWD')


class RecommenderModel:
    """
    Internal representation of the Recommender resource and helper.
    Client doesn't interact directly with these methods.
    """

    def __init__(self, job_ids=None, columns=None, feature_columns=None, feature_weights=None):
        if job_ids is None:
            self.job_ids = [555, 444]
        else:
            self.job_ids = [int(job_id) for job_id in job_ids.split()]
        self.sim_columns = [f'similarity_{job_id}' for job_id in self.job_ids]

        if columns is None:
            self.columns = ['id', 'title', 'company', 'remote', 'location', 'stack', 'text', 'experience', 'url']
        else:
            self.columns = columns

        self.res_columns = self.columns + self.sim_columns
        self.res_columns += ['mean_similarity']

        if feature_columns is None:
            self.feature_columns = ['remote', 'title', 'stack', 'text', 'experience']
        else:
            self.feature_columns = feature_columns

        if feature_weights is None:
            self.feature_weights = {
                'remote_similarity': 1,
                'title_similarity': 0.8,
                'stack_similarity': 0.8,
                'text_similarity': 0.7,
                'experience_similarity': 0.6,
            }
        else:
            self.feature_weights = feature_weights
        self.feature_names = None
        self.original_df = self.preprocess()

    def recommend(self):
        # if len(self.job_ids) > 1:
        for job_id in self.job_ids:
            # Get similarity_<job_id> Series
            similarity = self.compute_similarity(job_id)
            # Merge series with original dataframe
            self.original_df = self.original_df.merge(similarity, left_index=True, right_index=True)
        # Compute the mean of multiple <job_id>_similarity columns
        self.original_df = self.compute_mean_similarities(self.original_df)
        self.original_df.sort_values(by='mean_similarity', ascending=False, inplace=True)

    def compute_similarity(self, job_id):
        """Returns a Series of similarities for a given job."""
        # For one job id, computes the similarity of each given features separately
        res = self.compute_all_similarities(self.original_df, self.feature_columns, job_id)
        # Multiple each features with personalised weight
        df = self.compute_weighted_similarity(res, self.feature_weights)
        # Returns normalised column for given job_id
        return self.normalise_computed_weighted_similarity(df, job_id)

    def preprocess(self):
        df = self.extract_data(self.columns)
        self.fill_na(df, self.columns)
        df['stack'] = df.apply(self.strip_stack, axis=1)
        return df

    def extract_data(self, columns):
        engine = create_engine(f"postgresql://{USER}:{PWD}@localhost:5432/job_market")
        query = 'SELECT * FROM relevant;'
        relevant = pd.read_sql_query(query, engine)
        relevant = relevant[
            ['id', 'title', 'company', 'remote', 'location', 'stack', 'education', 'size', 'experience', 'url',
             'industry', 'type', 'created_at', 'text', 'summary']]

        user_df = relevant[['id']]
        item_df = relevant[columns]
        df = pd.merge(user_df, item_df, on='id')
        return df

    def fill_na(self, df, columns):
        for column in columns:
            df[column] = df[column].fillna('')

    def strip_stack(self, row):
        new_row = row['stack'].replace('{', '').replace('}', '').split(',')
        return new_row
        # return [w for w in row['stack']] # ast literal eval

    def combine_features(self, row, feature_column):
        # 'remote', 'title', 'stack', 'text', 'experience', 'size'
        new_row = ''
        if feature_column == 'stack':  # a list of words
            for w in row['stack']:
                new_row = new_row + ' ' + w
                return new_row
        elif feature_column == 'title' or feature_column == 'experience' or feature_column == 'size':
            return row[feature_column]
        elif feature_column == 'text':
            for w in [w for w in row['text'].split(' ')]:
                new_row = new_row + ' ' + w
                return new_row
        elif re.search(' +', row[feature_column]):  # multiple words
            for i in range(len(row[feature_column])):
                new_row = new_row + ' ' + str(row[feature_column[i]])
            return new_row
        elif not re.search(' +', row[feature_column]):  # only one word
            return row[feature_column]

    def extract_features(self, df):
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])
        return count_matrix

    def compute_individual_similarity(self, df, job_id, feature_column, feature_name):
        # Combine the features in one field
        df["combined_features"] = df.apply(lambda x: self.combine_features(x, feature_column), axis=1)

        # Compute the count matrix then similarities
        count_matrix = self.extract_features(df)
        cosine_sim = cosine_similarity(count_matrix)

        # Get similar jobs for one job_id
        similar_jobs = list(enumerate(cosine_sim[job_id]))

        # Keep similarity scores in a Series
        similarities = [similar_jobs[i][1] for i in range(len(similar_jobs))]
        return pd.Series(similarities).rename(feature_name)

    def compute_all_similarities(self, df, feature_columns, job_id):
        # Create base DataFrame indicating job_id that similarity is computed for
        individual_similarity = self.compute_individual_similarity(df=df, job_id=job_id,
                                                                   feature_column=self.feature_columns[0],
                                                                   # first column just to initiate the function
                                                                   feature_name='')
        individual_similarities = pd.DataFrame(index=individual_similarity.index)
        individual_similarities['job_id'] = job_id

        # Get features names
        self.feature_names = list()

        # For each feature, compute similarity in its own column
        for feature_column in feature_columns:
            feature_name = feature_column + '_similarity'
            self.feature_names.append(feature_name)
            individual_similarity = self.compute_individual_similarity(df=df, job_id=job_id,
                                                                       feature_column=feature_column,
                                                                       feature_name=feature_name)
            individual_similarities = individual_similarities.merge(individual_similarity, left_index=True,
                                                                    right_index=True)

        return df.merge(individual_similarities, left_index=True, right_index=True)

    def compute_weighted_similarity(self, res, feature_weights):
        weights = list(feature_weights.values())
        weighted = res[self.feature_names].apply(lambda x: x * weights, axis=1)
        res['weighted_similarity'] = weighted.apply(np.sum, axis=1)
        return res.sort_values(by='weighted_similarity', ascending=False)

    def normalise_computed_weighted_similarity(self, weighted_df, job_id):
        top = (weighted_df['weighted_similarity'] - min(weighted_df['weighted_similarity']))
        bot = (max(weighted_df['weighted_similarity']) - min(weighted_df['weighted_similarity']))
        weighted_df[f'similarity_{job_id}'] = top / bot
        return weighted_df[f'similarity_{job_id}']

    def compute_mean_similarities(self, df):
        df['mean_similarity'] = df[self.sim_columns].mean(axis=1)
        return df

    def get_id_from_index(self, df, index):
        return df[df.index == index]["id"].values[0]

    def format_json(self):
        # for i in range(len(self.sim_columns)):
        #     print(f'{self.sim_columns[i]}')

        return {
            'id': self.original_df['id'],
            'title': self.original_df['title'],
            'company': self.original_df['company'],
            'remote': self.original_df['remote'],
            'location': self.original_df['location'],
            'stack': self.original_df['stack'],
            # 'text': self.original_df['text'],
            'experience': self.original_df['experience'],
            # 'size': self.original_df['size'],
            'mean_similarity': self.original_df['mean_similarity']
        }


def main():
    # job_ids = [333, 444, 555]
    rec = RecommenderModel()
    rec.recommend()
    json = rec.format_json()
    # original_df = base_recommender.original_df
    # for job_id in job_ids:
    #     recommender = RecommenderModel(job_id=job_id)
    #     sim = recommender.compute_similarity()  # Adds column similarity
    #     original_df = original_df.merge(sim, left_index=True, right_index=True)
    # df = base_recommender.compute_mean_similarities(original_df, job_ids)
    # df = df.sort_values(by='mean_similarity', ascending=False)  # Will shuffle index
    # print(df)
    # print(df.iloc[333])
    # print(base_recommender.get_id_from_index(df, 333))


if __name__ == '__main__':
    main()
