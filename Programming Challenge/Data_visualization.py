import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('TrainOnMe.csv')

# Setting aesthetic parameters in one step.
sns.set(style="whitegrid")

# Iterate over each column in your dataframe (excluding non-numeric columns)
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure(figsize=(10,6))
    df = df.dropna()

    # Boxplot
    sns.boxplot(df[column])
    
    plt.title(f'Boxplot for {column}')
    plt.show()
    
    # Histogram
