import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class KRecommend:
    def __init__(self, k):
        self.k = k
        self.__vectorizer = TfidfVectorizer(stop_words="english", min_df=2)

    def fit(self, data, text_column):
        self.text_column = text_column
        self.train = data[self.text_column]
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)

    def fit_from_db(self, table_name, id_column, text_column, connection):
        self.text_column = text_column
        self.id_column = id_column
        self.connection = connection
        self.train = pd.read_sql_table(table_name, self.connection)
        self.train.set_index(self.id_column, inplace=True)
        self.train = self.train[self.text_column]
        self.train_index = self.train.index
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)
        print("KRecommender fitted successfully")

    def predict(self, text):
        self.text = [text]
        self.__vectorized_text = self.__vectorizer.transform(self.text)
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # sort,reverse the order and get the index with the top k similarity, neglecting the first
        top_k_indices = np.flip(similarity_values.argsort())[1 : self.k + 1]
        return {index: self.data[self.text_column][index] for index in top_k_indices}

    def predict_from_db(self, text):
        self.text = [text]
        self.__vectorized_text = self.__vectorizer.transform(self.text)
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # sort,reverse the order and get the index with the top k similarity, neglecting the first
        top_k_indices = np.flip(similarity_values.argsort())[1 : self.k + 1]
        return {self.train_index[i]: similarity_values[i] for i in top_k_indices}


"""
1)Error if an in is put into it
2)Error if thereare mjssing values
3)Error if it gasnt been trajned
4)Error if it wants to train on another type
4)Error if there is no table
"""
