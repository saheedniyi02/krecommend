# <a href="https://github.com/saheedniyi02/krecommend">krecommend</a>
A python package for creating content-based text recommender systems on pandas dataframes and SQLAlchemy tables.

The recommendations are gotten by using cosine similarity to get similar items to a requested item and the top k similar items are then recommended.
### Dependencies
krecommend requires the following dependencies:

<ul><li>Python</li>
<li>NumPy</li>
<li>SciPy</li>
<li>Scikit-learn</li>
<li>Pandas for dealing with dataframes</li>
<li>SQLAlchemy for dealing with SQL tables</li></ul>

## Installation
Installation can be done with `pip`:
```shell
$ pip install krecommend
```
## How to use 
The detailed examples can be found <a href="https://github.com/saheedniyi02/Test-krecommend">here</a>
#### For a pandas data frame.
#Provided with a simple dataframe with index "id" ,
text (string) columns "title" and "content","int" column "Views".

##### load the dataframe
```py
import pandas as pd
dataframe = pd.read_csv("file_path", index_col=0)
#set the id as the index
dataframe.set_index("id")
```
##### import,initialize and fit on a pandas dataframe
```py
recommender = KRecommend(k=2)
recommender.fit(dataframe, text_columns=["content","title"])

```
##### get recommendations.
```py
test_content="This is a test content"
test_title="This is a test title"
#the .predict method accepts lists only, even if the length is 1.
recommendations=recommender.predict(test=[test_content,test_title])
```

The returned recommendations is a simple python dictionary with length (k, the number of requested recommendations)\
Each key in the dictionary represents the index (value of the "id" in this case) of that particular
recommendation in the dataframe, while the value represents the similarity (in %),The items in the dictionary are arranged in descending order of the similarity.




#### For an SQL alchemy table.
##### A simple SQLAlchemy table (ensure you add items to your table)
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



```
###### import,initialize and fit on SQLAlchemy table
```py
#database connection
connection = engine.connect()
from krecommend.recommend import KRecommend
#k represents the number of documents to be recommend
recommender = KRecommend(k=4)
recommender.fit_on_sql_table(table_name="Posts",id_column= "id",text_columns=["content","title"],connection= connection)
#close connection
connection.close()
```

###### get recommendations.
```py
test_content="This is a test content"
test_title="This is a test title"
#the .predict_on_sql_table method accepts lists only, even if the length is 1.
recommendations=recommender.predict_on_sql_table(test=[test_content,test_title])
```
The returned recommendations is a simple python dictionary with length (k, the number of requested recommendations)\
Each key in the dictionary represents the primary_key of that particular
recommendation in the database, while the value represents the similarity (in %).The items in the dictionary are arranged in descending order of the similarity.



The primary key can then be used to query the table to get more information on the recommendations.



#### For a flask-sqlalchemy table
###### create the simple Flask-SQLAlchemy table (ensure you add items to your table)
```py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db = SQLAlchemy(app)

"""a table with name 'Posts', primary_key 'id', text (string) columns 'title' and 'content' and Int column 'views' """
class Posts(db.Model):
    __tablename__="Posts"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(64))
    content = db.Column(db.String(64))
    views = db.Column(db.Integer, unique=True, index=True, nullable=False)

```

###### import,initialize and fit on SQLAlchemy table
```py

from krecommend.recommend import KRecommend
#k represents the number of documents to be recommend
#database connection
connection=db.engine.connect()
recommender = KRecommend(k=4)
recommender.fit_on_sql_table(table_name="Posts",id_column= "id",text_columns=["content","title"],connection= connection)
#close connection
connection.close()
```
The recommendations can easily be gotten using the `.predict_on_sql_table` as seen above.

#### Warning and possible sources of error
<ol><li>Only text columns are accepted in the text_columns parameter.
Integer or float columns will return an error.</li><br>
<li>KRecommend only saves information on your table at the time it is fitted, any information on your table added after
KRecommend has been fitted won't exist in the recommendations generated.
<ul><li>Implications: <ol><li>A recommendation might have been deleted (after fitting) from the table as at the time it is being recommend so it might no longer be found in the database.</li>
              <li>Some content might have been modified which might affect the strength of the recommendations.<li></li></ol>
<li>Solution: it is important to fit KRecommend again at intervals,so changes in contents will be reflected in the recommendations.For example you can schedule  the `.fit_on_sql_table` method to run every hour (or any interval of your choice)</li></li></ul>
<li>It is good practice to close the connection after fitting.</li>
<li>There must be k+1 (k represents the requested no of recommendations) items in the requested table.</li>

