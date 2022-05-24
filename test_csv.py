import pandas as pd
from recommend import KRecommend


df = pd.read_csv("recommended_text.csv", index_col=0)
df = df.dropna()
recommender = KRecommend(k=4)
recommender.fit(df, text_column="content")
index = 419
print(df["content"][index])
print(recommender.predict(df["content"][index]))
