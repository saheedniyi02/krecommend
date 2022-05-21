import numpy as np
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd


df=pd.read_csv("recommended_text.csv",index_col=0)
df=df.dropna()

class KRecommend():
	"""
	A class for a document content based recommender system based on similarity of the documents.
	...
	Attributes
	__________
	k: int
		no of similar documents to be recommended
	text_column: string
	    name of the text column
	vectorizer :
		vectorizer used
		
	"""
	def __init__(self,k,text_column):
		"""
		
		"""
		self.k=k
		self.__vectorizer=CountVectorizer(stop_words="english",min_df=2)
		self.text_column=text_column
		
		
	def fit(self,data):
		self.train=data[self.text_column]
		self.__vectorized_train=self.__vectorizer.fit_transform(self.train)
		
		
	def predict(self,text):
		self.text=[text]
		self.__vectorized_text=self.__vectorizer.transform(self.text)
		similarity_values=cosine_similarity(self.__vectorized_text,self.__vectorized_train)[0]
		#sort,reverse the order and get the index with the top k similarity, neglecting the first
		top_k_indices=np.flip(similarity_values.argsort())[1:self.k+1]
		return [{index:df[self.text_column][index]} for index in top_k_indices]

		
"""
1)Error if an in is put into it
2)Error if thereare mjssing values
3)Error if it gasnt been trajned
"""
recommender=KRecommend(k=4,text_column="content")
recommender.fit(df)
index=419
print(df["content"][index])
print(recommender.predict(df["content"][index]))