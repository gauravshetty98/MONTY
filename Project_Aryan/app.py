import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import plotly.express as px
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
st.set_page_config(layout="wide")
# Title of the Streamlit App


st.markdown("<h1 style='text-align: center;'>Occupational Clustering and Analysis</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #332F28; padding: 20px; border-radius: 10px;">
        <h4 style="text-align: center; color: white;">Shifts in Job Demand</h4>
        <ul style="font-size:18px; color: white;">
            <li><strong>Manual Labor & Retail Roles (Cluster 0):</strong>
                <ul>
                    <li>While still dominant, their overall share might have shrunk.</li>
                    <li>Suggests potential impacts from automation, outsourcing, or reduced demand.</li>
                </ul>
            </li>
            <li><strong>High-Skilled and Managerial Roles (Cluster 1):</strong>
                <ul>
                    <li>Retain high salaries but employ a smaller workforce.</li>
                    <li>Increased presence of technology-related roles indicates growing demand for digital skills.</li>
                </ul>
            </li>
            <li><strong>Emerging Mid-Level Jobs (Cluster 2):</strong>
                <ul>
                    <li>The new cluster signifies a shift towards mid-salaried roles requiring a mix of skills.</li>
                    <li>Examples: Marketing Managers, Operations Specialists, Technical Repairers.</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #332F28; padding: 20px; border-radius: 10px;">
        <h4 style="text-align: center; color: white;">Key Insights on Changes (2019 → 2023)</h4>
        <ul style="font-size:18px; color: white;">
            <li><strong>Diverse Job Distribution:</strong> The 2023 workforce is more diverse, with a visible mid-level cluster.</li>
            <li><strong>Declining Manual Labor:</strong> The dominance of low-paying, manual-intensive jobs is waning.</li>
            <li><strong>Sustained Demand for Specialized Skills:</strong> Roles in technology, business management, and finance continue to be in high demand and command higher salaries.</li>
            <li><strong>Growth of Mid-Level Jobs:</strong> The emergence of mid-tier occupations indicates a shift towards valuing moderate skills.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)





# Read 2023 Data
data_2023 = pd.read_csv('national_M2023_dl.csv')
columns_to_use_2023 = ['OCC_CODE', 'OCC_TITLE', 'TOT_EMP', 'A_MEAN']
data_2023 = data_2023[columns_to_use_2023][1:]
data_2023['TOT_EMP'] = pd.to_numeric(data_2023['TOT_EMP'].replace({',': ''}, regex=True), errors='coerce')
data_2023['A_MEAN'] = pd.to_numeric(data_2023['A_MEAN'].replace({',': ''}, regex=True), errors='coerce')
data_2023 = data_2023.dropna()

# Read 2019 Data
data_2019 = pd.read_csv('national_M2019_dl.csv')
columns_to_use_2019 = ['occ_code', 'occ_title', 'tot_emp', 'a_mean']
data_2019 = data_2019[columns_to_use_2019][1:]
data_2019['tot_emp'] = pd.to_numeric(data_2019['tot_emp'].replace({',': ''}, regex=True), errors='coerce')
data_2019['a_mean'] = pd.to_numeric(data_2019['a_mean'].replace({',': ''}, regex=True), errors='coerce')
data_2019 = data_2019.dropna()

# Normalize Data and Apply KMeans Clustering for 2023
scaler = MinMaxScaler()
numerical_cols_2023 = ['TOT_EMP', 'A_MEAN']
data_scaled_2023 = scaler.fit_transform(data_2023[numerical_cols_2023])
kmeans_2023 = KMeans(n_clusters=2, random_state=42)
data_2023['Cluster'] = kmeans_2023.fit_predict(data_scaled_2023)
distances = pairwise_distances_argmin_min(data_scaled_2023, kmeans_2023.cluster_centers_)[1]
threshold = np.percentile(distances, 99)
data_2023 = data_2023[distances <= threshold]
data_scaled_2023 = scaler.fit_transform(data_2023[numerical_cols_2023])
kmeans = KMeans(n_clusters=3, random_state=42)
data_2023.loc[:, 'Cluster'] = kmeans.fit_predict(data_scaled_2023)


# Normalize Data and Apply KMeans Clustering for 2019
numerical_cols_2019 = ['tot_emp', 'a_mean']
data_scaled_2019 = scaler.fit_transform(data_2019[numerical_cols_2019])
kmeans_2019 = KMeans(n_clusters=2, random_state=42)
data_2019['Cluster'] = kmeans_2019.fit_predict(data_scaled_2019)
distances = pairwise_distances_argmin_min(data_scaled_2019, kmeans_2019.cluster_centers_)[1]
threshold = np.percentile(distances, 99)
data_2019 = data_2019[distances <= threshold]
data_scaled_2019 = scaler.fit_transform(data_2019[numerical_cols_2019])
kmeans_2019 = KMeans(n_clusters=2, random_state=42)
data_2019.loc[:, 'Cluster'] = kmeans_2019.fit_predict(data_scaled_2019)



st.markdown("""<h2 style="text-align: center; color: white;">Workforce Distribution Across Clusters</h2>""", unsafe_allow_html=True)
st.subheader("")

st.markdown("<h1 style='text-align: center;'>Top Occupations 2019</h1>", unsafe_allow_html=True)
top_occupations_2019 = data_2019.groupby('Cluster').apply(
    lambda x: x.nlargest(5, 'tot_emp')[['occ_title', 'tot_emp']]
)
col1, col2, col3 = st.columns([1.75, 2, 1.5])
with col2:
    st.dataframe(top_occupations_2019)
with col3:
    st.markdown("""
        <div style="font-size: 20px; background-color: #323232; padding: 10px; border-radius: 10px; color: white;">
            <h3 style="font-size: 22px; font-weight: bold;">2019:</h3>
            <ul style="font-size: 18px;">
                <li><strong style="font-size: 18px;">Cluster 0:</strong> High total employment, lower average salaries (manual labor, retail, support roles).</li>
                <li><strong style="font-size: 18px;">Cluster 1:</strong> Low total employment, higher average salaries (specialized and managerial roles).</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    st.subheader("")

# Top Occupations by Salary for 2019
st.markdown("<h1 style='text-align: center;'>Top Salaried 2019</h1>", unsafe_allow_html=True)
top_salaried_2019 = data_2019.groupby('Cluster').apply(
    lambda x: x.nlargest(5, 'a_mean')[['occ_title', 'a_mean']]
)

col1, col2, col3 = st.columns([2, 2, 1.5])
with col2:
    st.dataframe(top_salaried_2019)


# Scatterplot for 2019 Data
st.markdown("### Scatterplot for 2019 Data")
fig_2019 = px.scatter(
    data_2019,
    x='a_mean',
    y='tot_emp',
    color='Cluster',
    title="Occupational Clusters (2019)",
    labels={'a_mean': 'Average Salary', 'tot_emp': 'Total Employment'},
    hover_data=['occ_title']
)
st.plotly_chart(fig_2019)

# Log-Transformed Scatterplot for 2019 Data
st.markdown("### Log-Transformed Scatterplot for 2019 Data")
data_2019['Log_TOT_EMP'] = np.log1p(data_2019['tot_emp'])
data_2019['Log_A_MEAN'] = np.log1p(data_2019['a_mean'])
fig_2019_log = px.scatter(
    data_2019,
    x='Log_A_MEAN',
    y='Log_TOT_EMP',
    color='Cluster',
    title="Log-Transformed Occupational Clusters (2019)",
    labels={'Log_A_MEAN': 'Log(Average Salary)', 'Log_TOT_EMP': 'Log(Total Employment)'},
    hover_data=['occ_title']
)
st.plotly_chart(fig_2019_log)

# Top Occupations by Employment for 2023
st.markdown("<h1 style='text-align: center;'> Top Occupations by Employment (2023)</h1>", unsafe_allow_html=True)
top_occupations_2023 = data_2023.groupby('Cluster').apply(
    lambda x: x.nlargest(5, 'TOT_EMP')[['OCC_TITLE', 'TOT_EMP']]
)

col1, col2, col3 = st.columns([2, 3, 1.75])
with col2:
    st.dataframe(top_occupations_2023)
with col3:
    st.markdown("""
        <div style="background-color: #323232; padding: 20px; border-radius: 10px; color: white;">
            <h3 style="font-size: 22px; font-weight: bold;">2023:</h3>
            <ul style="font-size: 18px;">
                <li><strong style="font-size: 18px;">Cluster 0:</strong> Remains largely unchanged, dominated by labor-intensive roles.</li>
                <li><strong style="font-size: 18px;">Cluster 1:</strong> Continues to feature specialized, higher-paying jobs.</li>
                <li><strong style="font-size: 18px;">Cluster 2:</strong> Emerges as a new cluster representing emerging mid-level roles that balance employment size and salary.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

# Top Occupations by Salary for 2023
st.markdown("<h1 style='text-align: center;'> Top Occupations by Salary (2023)</h1>", unsafe_allow_html=True)
top_salaried_2023 = data_2023.groupby('Cluster').apply(
    lambda x: x.nlargest(5, 'A_MEAN')[['OCC_TITLE', 'A_MEAN']]
)
col1, col2, col3 = st.columns([2, 2.5, 1.75])
with col2:
    st.dataframe(top_salaried_2023)

# Scatterplot for 2023 Data
st.markdown("### Scatterplot for 2023 Data")
fig_2023 = px.scatter(
    data_2023,
    x='A_MEAN',
    y='TOT_EMP',
    color='Cluster',
    title="Occupational Clusters (2023)",
    labels={'A_MEAN': 'Average Salary', 'TOT_EMP': 'Total Employment'},
    hover_data=['OCC_TITLE']
)
st.plotly_chart(fig_2023)


# Log-Transformed Scatterplot for 2023 Data
st.markdown("### Log-Transformed Scatterplot for 2023 Data")
data_2023['Log_TOT_EMP'] = np.log1p(data_2023['TOT_EMP'])
data_2023['Log_A_MEAN'] = np.log1p(data_2023['A_MEAN'])
fig_2023_log = px.scatter(
    data_2023,
    x='Log_A_MEAN',
    y='Log_TOT_EMP',
    color='Cluster',
    title="Log-Transformed Occupational Clusters (2023)",
    labels={'Log_A_MEAN': 'Log(Average Salary)', 'Log_TOT_EMP': 'Log(Total Employment)'},
    hover_data=['OCC_TITLE']
)
st.plotly_chart(fig_2023_log)

# Create three columns, with the middle column` being the widest
col1, col2 = st.columns([3, 3])
with col1:
    st.markdown("""
    <div style="background-color: #332F28; padding: 10px 20px; border-radius: 10px; color: white; line-height: 1.4;">
        <h3 style="font-size: 25px; text-align: center; margin: 0;">2019 Insight Analysis</h3>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">Workforce Concentration</h4>
        <p style="margin: 5px 0; font-size: 18px;">A significant portion of the workforce (Cluster 0) is concentrated in lower-paying, labor-intensive jobs. This highlights economic sectors that rely heavily on large-scale employment but offer limited wage growth.</p>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">High-Skilled Roles</h4>
        <p style="margin: 5px 0;font-size: 18px;">Cluster 1 identifies specialized roles that are better compensated but employ a much smaller workforce. This suggests a divide in the labor market: occupations requiring advanced education or specialized skills offer higher salaries but fewer positions.</p>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">Labor Market Implications</h4>
        <p style="margin: 5px 0;font-size: 18px;">Cluster 0 jobs may be vulnerable to automation or economic downturns due to their lower specialization. Cluster 1 jobs highlight opportunities for career growth and salary optimization through education, upskilling, and specialization.</p>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">From 2019 to 2023:</h4>
        <ul style="margin: 5px 0; padding-left: 20px;font-size: 18px;">
            <li>The labor market has seen a shift away from purely manual roles towards mid-tier and specialized occupations.</li>
            <li>This highlights the growing importance of education, technical training, and adaptability to meet changing job demands.</li>
            <li>The emergence of mid-level roles creates opportunities for workers to transition from low-paying jobs into more stable, higher-paying roles with the right skills and training.</li>
            <li>This evolution reflects ongoing economic changes driven by technology, automation, and shifting workforce needs.</li>
        </ul>
    </div>""", unsafe_allow_html=True)
# Content to be displayed in the middle column inside a dark green box
with col2:
    st.markdown("""
    <div style="background-color: #0F3325; padding: 10px 20px; border-radius: 10px; color: white; line-height: 1.4;">
        <h3 style="font-size: 25px; text-align: center; margin: 0;">2023 Insight Analysis</h3>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">Cluster 0: High Employment, Low to Mid Skill Roles</h4>
        <p style="font-size:18px; text-align:justify;">This cluster includes occupations that have the highest number of workers, often found in industries like manual labor, retail, and healthcare support. 
        These jobs are typically lower-paying and require fewer specialized skills. While they are essential to the economy, they may be more vulnerable to automation and tend to offer fewer opportunities for upward mobility without additional training or education.</p>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">Cluster 1: Niche Roles with Lower Employment</h4>
        <p style="font-size:18px; text-align:justify;">Cluster 1 represents highly specialized and technical roles, including jobs in fields such as environmental science, technology, and specific mechanical trades. These roles tend to have lower total employment numbers 
        but are generally well-compensated due to the specialized skill sets required. While the labor market for these jobs is smaller, they offer higher salaries and greater job security for those with the right qualifications and experience.</p>
        <h4 style="font-size: 20px; font-weight: bold; margin: 10px 0 5px;">Cluster 2: Mid-to-High Employment, High-Skilled Jobs</h4>
        <p style="font-size:18px; text-align:justify;">Occupations in Cluster 2 are characterized by moderate to high employment and require advanced skills or management experience. These roles span across leadership, legal, and managerial professions, including positions 
        such as operations managers and financial managers. These jobs not only offer higher salaries but also provide career growth potential for individuals with strong expertise and leadership abilities. This cluster indicates a growing demand for skilled professionals in management and specialized fields.</p>
    </div>
    """, unsafe_allow_html=True)







