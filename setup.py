import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="krecommend",
    version="0.0.1",
    author="Saheed Azeez",
    author_email="saheedniyi02@gmail.com",
    description="A Python library for easily building  recommender systems for SQLAlchemy tables and pandas dataframe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saheedniyi02/KRecommend",
    keywords=[
        "Data Science",
        "machine learning",
        "web development",
        "software development",
        "sql",
        "pandas",
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        "pandas",
        "sqlalchemy",
        "numpy",
        "scikit-learn",
    ],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
    ],
)
