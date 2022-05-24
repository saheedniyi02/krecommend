from curses import meta
from pickle import TRUE
import pandas as pd
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Table

df = pd.read_csv("recommended_text.csv", index_col=0)
df = df.dropna()
engine = create_engine("sqlite:///database.db", echo=True)
meta = MetaData()

posts = Table(
    "Posts",
    meta,
    Column("id", Integer, primary_key=True),
    Column("title", String),
    Column("content", String),
)

meta.create_all(engine)
data = [{"title": df["title"][i], "content": df["content"][i]} for i in range(200)]
ins = posts.insert()

conn = engine.connect()
result = conn.execute(ins, data)
