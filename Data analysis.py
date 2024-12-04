# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(url, header=None, names=columns)

# Step 1: Compute basic statistics
basic_statistics = iris_df.describe()
print("Basic Statistics of Numerical Columns:")
print(basic_statistics)

# Step 2: Group by species and compute mean sepal length
mean_sepal_length_by_species = iris_df.groupby('species')['sepal_length'].mean().reset_index()
print("\nMean Sepal Length by Species:")
print(mean_sepal_length_by_species)

# Step 3: Identify patterns or interesting findings
print("\nAnalysis of Patterns:")
for index, row in mean_sepal_length_by_species.iterrows():
    print(f"The average sepal length for {row['species']} is {row['sepal_length']:.2f} cm.")
