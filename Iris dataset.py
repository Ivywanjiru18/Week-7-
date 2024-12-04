import pandas as pd

# Load the Iris dataset from a URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=columns)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Check data types and missing values
print("\nDataset information:")
print(iris_df.info())
print("\nMissing values in each column:")
print(iris_df.isnull().sum())

# Clean the dataset by dropping any rows with missing values (if there were any)
cleaned_iris_df = iris_df.dropna()

# Display cleaned data information
print("\nCleaned dataset information:")
print(cleaned_iris_df.info()
