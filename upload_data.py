import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Orange divider function
def orange_divider():
    st.markdown("<hr style='border:2px solid orange;'>", unsafe_allow_html=True)

# Function to generate a sample dataset for testing purposes
def generate_dataset():
    np.random.seed(42)
    demand = np.random.normal(loc=500, scale=50, size=1000)  # Simulating normal distribution for demand
    revenue = np.random.exponential(scale=300, size=1000)    # Simulating exponential distribution for revenue
    df = pd.DataFrame({
        'Demand': demand,
        'Revenue': revenue
    })
    return df

# Function to identify the best-fitting distribution
def find_best_distribution(data):
    distributions = {
        'Normal': stats.norm,
        'Exponential': stats.expon,
        'Poisson': stats.poisson
    }
    
    results = {}
    
    for name, dist in distributions.items():
        if name == 'Poisson':
            # For Poisson, we estimate the lambda (mean of the data, rounded)
            data = np.round(data)  # Poisson requires integer data
            param = np.mean(data)  # Poisson distribution is defined by its mean (lambda)
            ks_stat, p_value = stats.kstest(data, dist.cdf, args=(param,))
            params = (param,)
        else:
            # For normal and exponential distributions, use the `fit` method
            params = dist.fit(data)
            ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)
        
        results[name] = (ks_stat, p_value, params)
    
    # Choose the distribution with the highest p-value (best fit)
    best_fit = max(results.items(), key=lambda x: x[1][1])
    return best_fit

# Function to simulate data based on the best-fit distribution
def simulate_data(best_fit, size=1000):
    dist_name, (ks_stat, p_value, params) = best_fit
    dist = {
        'Normal': stats.norm,
        'Exponential': stats.expon,
        'Poisson': stats.poisson
    }[dist_name]
    
    if dist_name == 'Poisson':
        simulated_data = dist.rvs(params[0], size=size)
    else:
        simulated_data = dist.rvs(*params, size=size)
    
    return simulated_data

# Streamlit app layout
st.title("Demand/Revenue Data Simulation")
st.subheader("Upload your data and simulate based on best-fit distribution")
orange_divider()

# Dataset generation button
if st.button("Generate Sample Dataset"):
    df = generate_dataset()
    st.write(df)
    st.write("Download sample dataset:")
    st.download_button("Download CSV", df.to_csv(index=False), "sample_data.csv", "text/csv")

orange_divider()

# File upload
uploaded_file = st.file_uploader("Upload your demand/revenue data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

    # Select column for simulation
    column = st.selectbox("Select column for analysis", df.columns)

    # Find the best distribution for the selected data column
    data = df[column]
    best_fit = find_best_distribution(data)
    dist_name, (ks_stat, p_value, params) = best_fit

    st.write(f"Best-fit distribution: **{dist_name}**")
    st.write(f"Parameters: {params}")
    st.write(f"KS Statistic: {ks_stat:.4f}")
    st.write(f"P-Value: {p_value:.4f}")

    orange_divider()

    # Simulate data based on the best-fit distribution
    num_simulations = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)
    simulated_data = simulate_data(best_fit, size=num_simulations)

    # Plot the original and simulated data
    st.write(f"Simulated Data ({num_simulations} simulations):")
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot original data
    ax[0].hist(data, bins=30, color='lightblue', edgecolor='black', density=True)
    ax[0].set_title(f"Original Data ({column}) Distribution")
    ax[0].set_xlabel(column)
    ax[0].set_ylabel('Density')

    # Plot simulated data
    ax[1].hist(simulated_data, bins=30, color='lightgreen', edgecolor='black', density=True)
    ax[1].set_title(f"Simulated Data based on {dist_name} Distribution")
    ax[1].set_xlabel(f"Simulated {column}")
    ax[1].set_ylabel('Density')

    st.pyplot(fig)
