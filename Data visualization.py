import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=columns)

# 1. Line Chart (Simulated Trend)
iris_df['time'] = range(1, len(iris_df) + 1)
plt.figure(figsize=(10, 5))
plt.plot(iris_df['time'], iris_df['sepal_length'], marker='o', linestyle='-', color='b')
plt.title('Simulated Trend of Sepal Length Over Time')
plt.xlabel('Time')
plt.ylabel('Sepal Length (cm)')
plt.xticks(iris_df['time'])
plt.grid()
plt.show()

# 2. Bar Chart (Average Petal Length per Species)
mean_petal_length = iris_df.groupby('species')['petal_length'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal_length', data=mean_petal_length, palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=45)
plt.show()

# 3. Histogram (Distribution of Sepal Length)
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['sepal_length'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# 4. Scatter Plot (Sepal Length vs. Petal Length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=iris_df, palette='deep')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid()
plt.show()
