import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Caching the data for faster processing
@st.cache_data
def load_data():
    # Load Titanic data from scikit-learn
    dataset = fetch_openml("titanic", version=1, as_frame=True)
    data = dataset.frame
    # Drop rows with missing Age
    data = data.dropna(subset=["age"])
    return data

# Load data
data = load_data()

# Set the title
st.title("Titanic Dataset Analysis")

# Sidebar layout
st.sidebar.header("User Options")

# User Inputs
age_filter = st.sidebar.slider("Filter by Age", int(data['age'].min()), int(data['age'].max()), (0, 80))
gender_filter = st.sidebar.multiselect("Select Gender", ["male", "female"], default=["male", "female"])
class_filter = st.sidebar.selectbox("Select Passenger Class", ["All", 1, 2, 3])

# Filter data
filtered_data = data[
    (data['age'] >= age_filter[0]) &
    (data['age'] <= age_filter[1]) &
    (data['sex'].isin(gender_filter))
]
if class_filter != "All":
    filtered_data = filtered_data[filtered_data['pclass'] == class_filter]

# Dynamic Summary Statistic
st.subheader("Summary Statistic")
mean_age = filtered_data['age'].mean()
st.write(f"Mean age of filtered passengers: {mean_age:.2f} years")

# Layout with columns
col1, col2 = st.columns(2)
# Graphic 1: Survival Count by Gender
with col1:
    st.subheader("Survival Count by Gender")
    survival_count = filtered_data.groupby(['sex', 'survived']).size().unstack()
    survival_count.plot(kind="bar", stacked=True)
    plt.title("Survival Count by Gender")
    plt.ylabel("Count")
    plt.xlabel("Gender")
    st.pyplot(plt)

# Graphic 2: Age Distribution
with col2:
    st.subheader("Age Distribution")
    st.write("Adjust the bins for the histogram:")
    #4th user input
    bins = st.slider("Number of bins", 5, 50, 20)
    fig, ax = plt.subplots()
    filtered_data['age'].plot(kind="hist", bins=bins, ax=ax, color="skyblue", edgecolor="black")
    ax.set_title("Age Distribution of Passengers")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
