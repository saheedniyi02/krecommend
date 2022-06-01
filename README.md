# KRecommend
A python package for creating content-based text recommender systems on pandas dataframes and SQLAlchemy tables.

## Problem source and solution 
## Dependencies
Krecommend requires the following dependencies:

<ul><li>Python</li>
<li>NumPy</li>
<li>SciPy</li>
<li>Scikit-learn</li>
<li>Pandas for dealing with dataframes</li>
<li>SQLAlchemy for dealing with SQL tables</li></ul>

## Installation
```shell
$ pip install krecommend
```
## How to use
#### For a pandas data frame.
KRecommenm
###### Explain type of data frame.
###### Explain column names
###### Explain index
###### Explain returns


#### For an SQL alchemy table.
##### A simple SQLAlchemy table
```py
from curses import meta
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Table

#database engine
engine = create_engine("sqlite:///database.db", echo=True)
meta = MetaData()


"""a table with name 'Posts', primary_key 'id', text (string) columns 'title' and 'content' and Int column 'views' """
posts = Table(
    "Posts",
    meta,
    Column("id", Integer, primary_key=True),
    Column("title", String),
    Column("content", String),
    Column("views",Integer)
)


#database connection
connection = engine.connect()
```
###### import,initialize and fit on SQLAlchemy table
```py
from krecommend.recommend import KRecommend
#k represents the number of documents to be recommend
recommender = KRecommend(k=4)
recommender.fit_on_sql_table(table_name="Posts",id_column= "id",text_columns=["content","title"],connection= connection)
#close connection
connection.close()
```

###### get recommendations.
```py
new_content="This is a test content"
new_title="This is a test title"
recommendations=recommender.predict_on_sql_table(test=[new_content,new_title])
```




#### For a flask-sqlalchemy table
####### Explain difference.


#### Warning and possible sources of error

####
