import numpy as np
import pandas as pd

# Function to generate a sample dataset
def generate_sample_dataset():
    np.random.seed(42)
    demand = np.random.normal(loc=500, scale=50, size=1000)  # Normally distributed demand
    revenue = np.random.exponential(scale=300, size=1000)     # Exponentially distributed revenue
    df = pd.DataFrame({
        'Demand': demand,
        'Revenue': revenue
    })
    return df

# Generate and save the dataset as CSV
df = generate_sample_dataset()
df.to_csv("sample_demand_revenue_data.csv", index=False)
