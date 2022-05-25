import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import warnings
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class KRecommend:
    def __init__(self, k):
        self.k = k
        self.__vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
        self.__is_trained = False
        self.__is_trained_sql = False
        self.len_vocabulary=0

    def fit(self, data, text_columns):
        assert isinstance(text_columns,list),"'text_columns' parameter must be a list"
        self.text_columns = text_columns

        # handles 1 column or multiple columns
        if len(self.text_columns) == 1:
            try:
                #checks if column exists,creates the train data and raises an exception if a column not of type "string" was passed
                self.train = data[self.text_columns[0]]
            except KeyError:
                raise Exception("Column cannot be found in the dataframe")
            if ( data[self.text_columns[0]].dtype!="object"):
                raise Exception(f"{self.text_columns[0]} column isn't' a string column, pass a column of type string.")
        elif len(self.text_columns) > 1:
            self.train = ""
            for column in self.text_columns:
                if column not in data.columns:
                	raise Exception(f"{column} column cannot be found in dataframe")
                #checks if column exists,creates the train data and gives a warning if one of the columns were of type "int".
                if ( data[column].dtype!="object"):
                	data[column]=data[column].astype("object")
                	warnings.warn(f"{column} column isn't' a string column, you might want to change it to type string")
                self.train += self.train + data[column]+ " "
        self.train.fillna("",inplace=True)
        self.train_index = self.train.index
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)

        # records if the model has been trained or not and identifies which type of training
        self.__is_trained = True
        self.__is_trained_sql = False
        self.len_vocabulary=len(self.__vectorizer.vocabulary_)
        print(f"KRecommender fitted successfully with {self.len_vocabulary} words in the vocabulary!")

    def fit_on_sql_table(self, table_name, id_column, text_columns, connection):
        assert isinstance(text_columns,list),"'text_columns'' parameter must be a list"
        self.text_columns = text_columns
        self.id_column = id_column
        self.table_name=table_name
        self.connection = connection
        
        #table to dataframe
        data = pd.read_sql_table(table_name, self.connection)
        
        #set index and returns an error if id_column doesn't exist'
        try:
        	data.set_index(self.id_column, inplace=True)
        except KeyError:
        	raise Exception(f"{self.id_column} column cannot be found in the {self.table_name} table")     
        # handles 1 column or multiple columns
        if len(self.text_columns) == 1:
            try:
            	self.train = data[self.text_columns[0]]
            except KeyError:
            	raise Exception(f"Column cannot be found in {self.table_name} table")
            if ( data[self.text_columns[0]].dtype!="object"):
                raise Exception(f"{self.text_columns[0]} column isn't' a string column, pass a column of type string.")
        elif len(self.text_columns) > 1:
            self.train = ""
            for column in self.text_columns:
                if column not in data.columns:
                	raise Exception(f"{column} column cannot be found in {self.table_name} table") 	
                if ( data[column].dtype!="object"):
                	data[column]=data[column].astype("object")
                	warnings.warn(f"{column} column isn't' a string column, you might want to change it to type string")
                self.train += self.train + data[column]+ " "
        self.train.fillna("",inplace=True) 
        self.train_index = self.train.index
        self.__vectorized_train = self.__vectorizer.fit_transform(self.train)

        # records if the model has been trained or not and identifies which type of training
        self.__is_trained_sql = True
        self.__is_trained = False
        self.len_vocabulary=len(self.__vectorizer.vocabulary_)
        print(f"KRecommender fitted successfully with {self.len_vocabulary} words in the vocabulary!")

    def predict(self, test):
        assert isinstance(test,list),"'test'' parameter must be a list"
        assert len(test)==len(self.text_columns),f"KRecommend was trained on {len(self.text_columns)} columns while your input has {len(test)} columns"
        if self.__is_trained == False:
            raise Exception(
                "KRecommend model hasn't been trained on a pandas dataframe."
            )

        if len(test) == 1:
            self.test = test
        else:
            self.test = [" ".join(test)]
        self.__vectorized_text = self.__vectorizer.transform(self.test)
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # sort,reverse the order and get the index with the top k similarity, neglecting the first
        top_k_indices = np.flip(similarity_values.argsort())[0 : self.k + 1]
        recommendations={self.train_index[index]: round(similarity_values[index]*100,3) for index in top_k_indices}
        max_id=max(recommendations,key=recommendations.get)
        min_id=min(recommendations,key=recommendations.get)
        if recommendations[max_id]>98:
        	recommendations.pop(max_id)
        else:
        	recommendations.pop(min_id)
        	
        print(max_id,min_id)
        return recommendations
        

    def predict_on_sql_table(self, test):
        assert isinstance(test,list),"'test'' parameter must be a list"   
        assert len(test)==len(self.text_columns),f"KRecommend was trained on {len(self.text_columns)} columns while your input has {len(test)} columns"
        if self.__is_trained_sql == False:
            raise Exception("KRecommend model hasn't been trained on an sql table.")
        if len(test) == 1:
            self.test = test
        else:
            self.test = [" ".join(test)]
        self.__vectorized_text = self.__vectorizer.transform(self.test)
        similarity_values = cosine_similarity(
            self.__vectorized_text, self.__vectorized_train
        )[0]
        # sort,reverse the order and get the index with the top k similarity, neglecting the first
        top_k_indices = np.flip(similarity_values.argsort())[0: self.k + 1]
        recommendations= {self.train_index[index]:round(similarity_values[index]*100,3) for index in top_k_indices}
        max_id=max(recommendations,key=recommendations.get)
        min_id=min(recommendations,key=recommendations.get)
        if recommendations[max_id]>98:
        	recommendations.pop(max_id)
        else:
        	recommendations.pop(min_id)
        return recommendations


"""
1)Error if an in is put into it.Done
2)Error if there are missing values.Done
3)Error if it hasnt been trained.Done
4)Error if it wants to train on another type.Done
5)Error if there is no table.Done
6)Error if text column or id column can't be found.Done
7)Ensure the inputs are lists.Done
8)Print number of words in vocabulary.Done
9)Remove first one if it is 1.Done
future
1)additional cleaning
2)filter
3)Preference
"""
