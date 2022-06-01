import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def clean_data(text):
    text = text.replace("\n", "")
    return text


class KRecommend:
    """
    A python class for creating text content-based recommendation system for SQLAlchemy tables and pandas dataframes by computing the
    cosine similarity of documents in the table or dataframe.


    Parameters
    __________
    k: int,default=2
        number of documents to recommend.




    """

    def __init__(self, k=4):
        self.k = k
        self.__vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
        self.__is_trained = False
        self.__is_trained_sql = False
        self.len_vocabulary = 0

    def fit(self, data, text_columns):
        """
        Build a KRecommend on a pandas dataframe

        Parameters
        __________
        data: a pandas dataframe
            the dataframe for the model to be trained on

        text_columns: list
            a list of columns the similarity should be calculated on,ensure the column has a type "string"

        Returns
        _______

        self : KRecommend
            Fitted KRecommend

        """
        # checks if the text_columns is a list
        assert isinstance(text_columns, list), "'text_columns' parameter must be a list"
        self.text_columns = text_columns

        # for 1 text_column
        if len(self.text_columns) == 1:
            # checks if column exists,creates the train data and raises an exception if a column not of type "string" was passed
            try:
                self.train = data[self.text_columns[0]]
            except KeyError:
                raise Exception("Column cannot be found in the dataframe")
            if data[self.text_columns[0]].dtype != "object":
                raise Exception(
                    f"{self.text_columns[0]} column isn't' a string column, pass a column of type string."
                )

        # for multiple columns
        elif len(self.text_columns) > 1:
            self.train = ""
            for column in self.text_columns:
                if column not in data.columns:
                    raise Exception(f"{column} column cannot be found in dataframe")
                # checks if column exists,creates the train data and gives a warning if one of the columns is not of type "string".
                if data[column].dtype != "object":
                    data[column] = data[column].astype("object")
                    warnings.warn(
                        f"{column} column isn't' a string column, you might want to change it to type string"
                    )
                self.train += self.train + data[column] + " "

        # fill missing values
        self.train.fillna("", inplace=True)
        self.train = self.train.apply(clean_data)
        # get index
        self.train_index = self.train.index
        # fit the vectorizer on the train set and transform the train test.
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)

        # records if the model has been trained or not and identifies which type of training
        self.__is_trained = True
        self.__is_trained_sql = False
        self.len_vocabulary = len(self.__vectorizer.vocabulary_)
        print(
            f"KRecommender fitted successfully with {self.len_vocabulary} words in the vocabulary!"
        )
        return self

    def fit_on_sql_table(self, table_name, id_column, text_columns, connection):
        """
        Build a KRecommend on an SQLAlchemy table

        Parameters
        __________
        table_name: String
            the name of the table you wan to fit the model on

        id_columns: String
           name of the primary_key of the table

        text_columns: list
            a list of columns the similarity should be calculated on,ensure the column has a type "string"

        connection: an SQLAlchemy connection
            the SQLAlchemy connection of the database "engine.connect()" ,the connection should be closed after KRecommend has been fitted

        Returns
        _______

        self : KRecommend
            Fitted KRecommend

        """
        # checks if the text_columns is a list
        assert isinstance(
            text_columns, list
        ), "'text_columns'' parameter must be a list"
        self.text_columns = text_columns
        self.id_column = id_column
        self.table_name = table_name
        self.connection = connection

        # table to dataframe
        data = pd.read_sql_table(table_name, self.connection)

        # set index and returns an error if id_column doesn't exist'
        try:
            data.set_index(self.id_column, inplace=True)
        except KeyError:
            raise Exception(
                f"{self.id_column} column cannot be found in the {self.table_name} table"
            )
        # for 1 column
        if len(self.text_columns) == 1:
            try:
                self.train = data[self.text_columns[0]]
            except KeyError:
                raise Exception(f"Column cannot be found in {self.table_name} table")
            if data[self.text_columns[0]].dtype != "object":
                raise Exception(
                    f"{self.text_columns[0]} column isn't' a string column, pass a column of type string."
                )

        # for multiple columns
        elif len(self.text_columns) > 1:
            self.train = ""
            for column in self.text_columns:
                if column not in data.columns:
                    raise Exception(
                        f"{column} column cannot be found in {self.table_name} table"
                    )
                if data[column].dtype != "object":
                    data[column] = data[column].astype("object")

                    warnings.warn(
                        f"{column} column isn't' a string column, you might want to change it to type string"
                    )
                self.train += self.train + data[column] + " "
        self.train.fillna("", inplace=True)
        self.train = self.train.apply(clean_data)
        self.train_index = self.train.index
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)

        # records if the model has been trained or not and identifies which type of training
        self.__is_trained_sql = True
        self.__is_trained = False
        self.len_vocabulary = len(self.__vectorizer.vocabulary_)
        print(
            f"KRecommender fitted successfully with {self.len_vocabulary} words in the vocabulary!\nEnsure you close your sql connection!"
        )
        return self

    def predict(self, test):
        """Predict on the input sample. Shoukd only be used if KRecommend was trained on a pandas dataframe.

        Parameters
        __________
        test: list
            list of features of the input for recommendations to be generated for, it is necessary for it to have same length as the
            "text_columns".
        Returns
        _______
        recommendations : Dict
            a dictionary of length k (number of documents to recommend),each key represents the id of a recommendation in the dataframe and
            the value represents the similarity in percentage

        """

        # asserts that input is a list and has same length as train data
        assert isinstance(test, list), "'test'' parameter must be a list"
        assert len(test) == len(
            self.text_columns
        ), f"KRecommend was trained on {len(self.text_columns)} columns while your input has {len(test)} columns"
        if self.__is_trained == False:
            raise Exception(
                "KRecommend model hasn't been trained on a pandas dataframe."
            )

        if len(test) == 1:
            self.test = test
        else:
            self.test = [" ".join(test)]
        self.test = [clean_data(self.test[0])]
        self.__vectorized_text = self.__vectorizer.transform(self.test)
        # calculate similarity
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # get the index with the top k similarity
        top_k_indices = np.flip(
            similarity_values.argsort()[len(similarity_values) - self.k - 1 :]
        )
        recommendations = {
            self.train_index[index]: round(similarity_values[index] * 100, 3)
            for index in top_k_indices
        }
        max_id = max(recommendations, key=recommendations.get)
        min_id = min(recommendations, key=recommendations.get)
        if recommendations[max_id] > 98:
            recommendations.pop(max_id)
        else:
            recommendations.pop(min_id)

        return recommendations

    def predict_on_sql_table(self, test):
        """Predict on the input sample. Shoukd only be used if KRecommend was trained on an SQLAlchemy table
        
        Parameters
        __________
        test: list
            list of features of the input for recommendations to be generated for, it is necessary for it to have same length as the
            "text_columns".
        Returns
        _______
        recommendations : Dict
            a dictionary of length k (number of documents to recommend),each key represents the primary_key of a recommendation in the SQLAlchemy table and the value represents the similarity in percentage.

        """
        assert isinstance(test, list), "'test'' parameter must be a list"
        assert len(test) == len(
            self.text_columns
        ), f"KRecommend was trained on {len(self.text_columns)} columns while your input has {len(test)} columns"
        if self.__is_trained_sql == False:
            raise Exception("KRecommend model hasn't been trained on an sql table.")
        if len(test) == 1:
            self.test = test
        else:
            self.test = [" ".join(test)]
        self.test = [clean_data(self.test[0])]
        self.__vectorized_text = self.__vectorizer.transform(self.test)
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # sort,reverse the order and get the index with the top k similarity, neglecting the first
        top_k_indices = np.flip(
            similarity_values.argsort()[len(similarity_values) - self.k - 1 :]
        )
        recommendations = {
            self.train_index[index]: round(similarity_values[index] * 100, 3)
            for index in top_k_indices
        }
        max_id = max(recommendations, key=recommendations.get)
        min_id = min(recommendations, key=recommendations.get)
        if recommendations[max_id] > 98:
            recommendations.pop(max_id)
        else:
            recommendations.pop(min_id)
        return recommendations
        
        #the .predict and .predict_on_sql_table methods are the same for now, they have been seperated so as to develop and add new features to them independently