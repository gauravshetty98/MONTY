import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import os

# Load the Excel file
file_path = 'education.xlsx'
table_5_2 = pd.read_excel(file_path, sheet_name='Table 5.2', skiprows=1)[1:]
table_5_4 = pd.read_excel(file_path, sheet_name='Table 5.4', skiprows=1)

# Preprocess the data
table_5_2.columns = [
    'Typical_Entry_Level_Education',
    'Employment_2019',
    'Employment_Distribution_2019',
    'Employment_Change_Percent_2019_29',
    'Median_Annual_Wage_2020'
]
table_5_2 = table_5_2.dropna()
table_5_4.columns = [
    'Job_Title',
    'Code',
    'Typical_Education',
    'Work_Experience',
    'On_The_Job_Training'
]
table_5_4['Work_Experience'] = table_5_4['Work_Experience'].fillna('No prior experience')
table_5_4 = table_5_4.drop(columns='On_The_Job_Training')
table_5_4 = table_5_4.dropna()

merged_data = pd.merge(
    table_5_2,
    table_5_4,
    how='inner',
    left_on='Typical_Entry_Level_Education',
    right_on='Typical_Education'
)

# Functions for preprocessing and recommendations
def preprocess_job_data(df):
    df.loc[:, 'Salary'] = df['Median_Annual_Wage_2020'].apply(lambda x: f"salary_{int(x // 10000)}k")
    df.loc[:, 'Combined_Features'] = (df['Typical_Education'].fillna('') + ' ' +
                                      df['Work_Experience'].fillna('') + ' ' +
                                      df['Salary'])
    return df

def filter_salary_data(df, desired_salary):
    salary_max = desired_salary + 15000
    salary_min = desired_salary - 15000
    return df[(df['Median_Annual_Wage_2020'] >= salary_min) & (df['Median_Annual_Wage_2020'] <= salary_max)]

def build_recommendation_system(df, user_education, desired_salary):
    salary_filtered_df = filter_salary_data(df, desired_salary)

    if salary_filtered_df.empty:
        st.warning("No jobs meet your desired salary. You may need to lower your salary expectations.")
        return pd.DataFrame()

    salary_filtered_df = preprocess_job_data(salary_filtered_df)

    user_query = {'Typical_Education': user_education,
                  'Work_Experience': 'No prior experience',
                  'Median_Annual_Wage_2020': desired_salary,
                  'Salary': f"salary_{int(desired_salary // 10000)}k",
                  'Combined_Features': f"{user_education} No prior experience salary_{int(desired_salary // 10000)}k"}

    salary_filtered_df = pd.concat([salary_filtered_df, pd.DataFrame([user_query])], ignore_index=True)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(salary_filtered_df['Combined_Features'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    user_similarity_scores = cosine_sim[-1][:-1]

    salary_filtered_df['Similarity'] = user_similarity_scores.tolist() + [1.0]
    recommendations = salary_filtered_df[:-1].sort_values(by='Similarity', ascending=False)

    if recommendations.empty:
        st.warning("No jobs match your education level. Consider obtaining higher qualifications.")
        return pd.DataFrame()

    return recommendations[['Job_Title', 'Median_Annual_Wage_2020', 'Work_Experience']].head(5)

# Streamlit UI
st.title("Career Recommendation System")

menu_options = [
    "Doctoral or professional degree",
    "Master's degree",
    "Bachelor's degree",
    "Associate's degree",
    "Postsecondary nondegree award",
    "Some college, no degree",
    "High school diploma or equivalent",
    "No formal educational credential"
]

user_education = st.selectbox("Select your education level", menu_options)
desired_salary = st.number_input("Enter your desired minimum salary (in USD)", min_value=0, step=1000)

if st.button("Generate"):
    recommendations = build_recommendation_system(merged_data, user_education, desired_salary)
    if not recommendations.empty:
        st.subheader("Top 5 Recommended Careers:")
        st.table(recommendations)

FEEDBACK_FILE = "Feedback_File.csv"

# Function to initialize the feedback file
def initialize_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        # Create the file with headers if it doesn't exist
        pd.DataFrame(columns=["Rating", "Improvement Suggestions"]).to_csv(FEEDBACK_FILE, index=False)
    elif os.path.getsize(FEEDBACK_FILE) == 0:  # Check if the file is empty
        # If the file is empty, ensure it has headers
        pd.DataFrame(columns=["Rating", "Improvement Suggestions"]).to_csv(FEEDBACK_FILE, index=False)

# Function to log feedback
def log_feedback(rating, suggestions=""):
    feedback_data = pd.read_csv(FEEDBACK_FILE)

    # Check if the DataFrame is empty and reset the column names if needed
    if feedback_data.empty:
        feedback_data = pd.DataFrame(columns=["Rating", "Improvement Suggestions"])

    new_entry = pd.DataFrame({"Rating": [rating], "Improvement Suggestions": [suggestions]})

    # Use pd.concat instead of append to add the new entry
    feedback_data = pd.concat([feedback_data, new_entry], ignore_index=True)
    feedback_data.to_csv(FEEDBACK_FILE, index=False)

# Initialize the feedback file
initialize_feedback_file()

# Streamlit UI
st.title("Career Recommendation System")

# Feedback Section
st.header("Feedback")

st.write("We'd love to hear your thoughts on this tool!")

# User selects feedback
feedback = st.radio("How would you rate your experience using this system?", ["Excellent", "Good", "Average", "Poor"], index=None)

# Additional input for 'Average' or 'Poor' ratings
improvement_reason = ""
if feedback in ["Average", "Poor"]:
    improvement_reason = st.text_area("How can we improve your experience?")

# Submit button
if st.button("Submit Feedback"):
    if feedback:
        log_feedback(feedback, improvement_reason)
        st.success("Thank you for your feedback! It has been logged successfully.")
        st.write(f"**Your Rating:** {feedback}")
        if feedback in ["Average", "Poor"]:
            st.write(f"**Improvement Suggestions:** {improvement_reason}")
    else:
        st.warning("Please provide a rating before submitting feedback.")

# Display logged feedback (optional: for admin review)
if st.checkbox("Show Logged Feedback (for Admin Review)"):
    feedback_data = pd.read_csv(FEEDBACK_FILE)
    st.dataframe(feedback_data)