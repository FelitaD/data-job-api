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
    Internal representation of the Job resource and helper.
    Client doesn't interact directly with these methods.
    """

    def __init__(self, job_id, columns=None, feature_columns=None, feature_names_weights=None):
        self.job_id = job_id
        if columns is None:
            self.columns = ['id', 'title', 'company', 'remote', 'location', 'stack', 'text', 'experience', 'size']
        else:
            self.columns = columns
        if feature_columns is None:
            self.feature_columns = ['remote', 'title', 'stack', 'text', 'experience', 'size']
        else:
            self.feature_columns = feature_columns
        if feature_names_weights is None:
            self.feature_names_weights = {
                'remote_similarity': 1,
                'title_similarity': 0.8,
                'stack_similarity': 0.8,
                'text_similarity': 0.7,
                'experience_similarity': 0.6,
                'size_similarity': 0.5
            }
        else:
            self.feature_names_weights = feature_names_weights
        self.feature_names = None
        self.original_df = self.preprocess()

    def recommend(self, job_id):
        sim = self.compute_similarity()  # Returns similarity series
        original_df = self.original_df.merge(sim, left_index=True, right_index=True)
        df = original_df.compute_mean_similarities(original_df, job_id)
        return df.sort_values(by='mean_similarity', ascending=False)

    def compute_similarity(self):
        """Returns a Series of similarities for a given job."""
        res = self.compute_all_similarities(self.original_df, self.feature_columns, self.job_id)
        df = self.compute_weighted_similarity(res, self.feature_names_weights)
        return self.normalise_computed_weighted_similarity(df)

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
            ['id', 'title', 'company', 'remote', 'location', 'stack', 'education', 'size', 'experience', 'rank', 'url',
             'industry', 'type', 'created_at', 'text', 'summary']]
        # print(relevant.info())

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
        new_row = ''
        if feature_column == 'stack':  # a list of words
            # print('list of words\n')
            for w in row['stack']:
                new_row = new_row + ' ' + w
                return new_row
        elif feature_column == 'title' or feature_column == 'experience' or feature_column == 'size':
            # print('title\n')
            return row[feature_column]
        elif feature_column == 'text':
            for w in [w for w in row['text'].split(' ')]:
                new_row = new_row + ' ' + w
                return new_row
        elif re.search(' +', row[feature_column]):  # multiple words
            # print('multiple words\n')
            for i in range(len(row[feature_column])):
                new_row = new_row + ' ' + str(row[feature_column[i]])
            return new_row
        elif not re.search(' +', row[feature_column]):  # only one word
            # print(f'one word: {row[feature_column]}\n')
            return row[feature_column]

    def extract_features(self, df):
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])
        # print(count_matrix.shape)
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

    def compute_weighted_similarity(self, res, feature_names_weights):
        weights = list(feature_names_weights.values())
        weighted = res[self.feature_names].apply(lambda x: x * weights, axis=1)
        res['weighted_similarity'] = weighted.apply(np.sum, axis=1)
        return res.sort_values(by='weighted_similarity', ascending=False)

    def normalise_computed_weighted_similarity(self, weighted_df):
        top = (weighted_df['weighted_similarity'] - min(weighted_df['weighted_similarity']))
        bot = (max(weighted_df['weighted_similarity']) - min(weighted_df['weighted_similarity']))
        weighted_df[f'similarity_{self.job_id}'] = top / bot
        return weighted_df[f'similarity_{self.job_id}']

    def compute_mean_similarities(self, df, job_ids):
        sim_columns = [f'similarity_{job_id}' for job_id in job_ids]
        df['mean_similarity'] = df[sim_columns].mean(axis=1)
        return df

    def get_id_from_index(self, df, index):
        return df[df.index == index]["id"].values[0]


def main():
    job_ids = [333, 444, 555]
    base_recommender = RecommenderModel(job_id=job_ids[0])
    original_df = base_recommender.original_df
    for job_id in job_ids:
        recommender = RecommenderModel(job_id=job_id)
        sim = recommender.compute_similarity()  # Adds column similarity
        original_df = original_df.merge(sim, left_index=True, right_index=True)
    df = base_recommender.compute_mean_similarities(original_df, job_ids)
    # df = df.sort_values(by='mean_similarity', ascending=False)  # Will shuffle index
    print(df)
    print(df.iloc[333])
    print(base_recommender.get_id_from_index(df, 333))


if __name__ == '__main__':
    main()
