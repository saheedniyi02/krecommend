import pandas as pd
from krecommend.recommend import KRecommend


df = pd.read_csv("recommended_text.csv", index_col=0)
df = df.dropna()
recommender = KRecommend(k=2)
recommender.fit(df, text_columns=["content"])
index = 421
print(df["content"][index])
print(recommender.predict([df["content"][index]]))
