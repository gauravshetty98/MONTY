import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from linkedin_api import Linkedin
from transformers import AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import plotly.graph_objects as go


def get_linkedin_profile(email, password, profile_id):
    # Initialize the LinkedIn API with email and password
    api = Linkedin(email, password)
    
    # Fetch the profile based on profile ID
    profile = api.get_profile(profile_id)
    
    # Return the profile information
    return profile


def initialize_dataframes():
    # Reading the 2023 data
    data_2023 = pd.read_excel('project_data/oesm23nat/national_M2023_dl.xlsx')
    
    # Reading the 2019 data
    data_2019 = pd.read_excel('project_data/oesm19nat/national_M2019_dl.xlsx')
    
    # Reading the 2033 industry data (from a specific sheet and skipping rows)
    ind_data_2033 = pd.read_excel('project_data/2023-33/industry.xlsx', sheet_name=11, skiprows=1)

    educ_data_2023 = pd.read_excel('project_data/2023-33/education.xlsx',sheet_name = 3, skiprows = 1)

    chg_proj_2019 = pd.read_excel('project_data/2019-29/occupation.xlsx', sheet_name = 8, skiprows = 1)
    chg_proj_2023 = pd.read_excel('project_data/2023-33/occupation.xlsx', sheet_name = 10, skiprows = 1)
    
    # Returning the dataframes as a dictionary
    return data_2023, data_2019, ind_data_2033, educ_data_2023, chg_proj_2019, chg_proj_2023


# from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
# model = AutoModelForMaskedLM.from_pretrained("jjzha/jobbert-base-cased")

def init_model():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
    model = AutoModel.from_pretrained("jjzha/jobbert-base-cased")
    return tokenizer, model


# Define mean pooling function
def mean_pooling(token_embeddings, attention_mask):
    # Multiply token embeddings by their respective attention mask
    token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
    # Sum embeddings across the sequence dimension and divide by the attention mask sum
    return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)


#ind_title = profile['industryName']
#data = data_2023['OCC_TITLE'
def get_similar_items(ind_title, col_vals, tokenizer, model):
    # Tokenize and get embeddings for the industry title
    ind_inputs = tokenizer(ind_title, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        ind_outputs = model(**ind_inputs)
        
    # Compute the industry title embedding
    ind_embedding = mean_pooling(ind_outputs.last_hidden_state, ind_inputs['attention_mask'])
    ind_vector = ind_embedding.squeeze().cpu().numpy()
    
    # List to store similarity scores and occupation titles
    similarities = []
    
    # Iterate through occupation titles and calculate similarity
    for occ_title in col_vals:
        # Tokenize inputs
        occ_inputs = tokenizer(occ_title, return_tensors="pt", truncation=True, padding=True)
        
        # Get embeddings from the model
        with torch.no_grad():
            occ_outputs = model(**occ_inputs)
        
        # Compute the occupation title embedding
        occ_embedding = mean_pooling(occ_outputs.last_hidden_state, occ_inputs['attention_mask'])
        occ_vector = occ_embedding.squeeze().cpu().numpy()
        
        # Compute cosine similarity
        similarity = 1 - cosine(ind_vector, occ_vector)
        
        # Store similarity and occupation title
        similarities.append((occ_title, similarity))
    
    
    # Sort by similarity in descending order and get the top 5
    sim_list = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    return list({title: score for title, score in sim_list}.items())


def single_bar_plot(sim_occs_list, data, col_1, label_1, title_col, plt_title):
    # Filter data for relevant occupations
    occs = set([title[0] for title in sim_occs_list])
    salary_data = data[data[title_col].isin(occs)].drop_duplicates(subset=[title_col], keep='first')
    
    # Extract data for plotting
    hourly_sals = list(salary_data[col_1])
    occ_titles = list(salary_data[title_col])
    
    # Create the figure
    fig = go.Figure()
    
    # Add bars for hourly salaries
    fig.add_trace(go.Bar(
        x=occ_titles,
        y=hourly_sals,
        name=label_1,
        marker_color='skyblue'
    ))
    
    # Update layout
    fig.update_layout(
        title=plt_title,
        xaxis=dict(
            title='Occupation Titles',
            tickangle=45,
            automargin=True,
        ),
        yaxis=dict(
            title=label_1,
            titlefont=dict(color='skyblue'),
            tickfont=dict(color='skyblue')
        ),
        height=600,  # Increase plot height
        width=1000   # Increase plot width
    )
    
    # Show the plot
    return fig


def double_bar_plot(sim_occs_list, data_1, data_2, col_1, col_2, label_1, label_2, title_col_1, title_col_2, plt_title):
    """
    Create a bar chart comparing two datasets by adding a second bar trace
    to the figure returned by single_bar_plot.

    Parameters:
        sim_occs_list (list): List of similar occupations.
        data_2020 (DataFrame): Data for 2020.
        data_2023 (DataFrame): Data for 2023.
        col_1_2020 (str): Column name for 2020 salaries.
        col_1_2023 (str): Column name for 2023 salaries.
        label_2020 (str): Label for 2020 data.
        label_2023 (str): Label for 2023 data.
        plt_title (str): Title of the plot.

    Returns:
        plotly.graph_objects.Figure: The generated figure.
    """
    # Get the single bar plot for 2020 data
    fig = single_bar_plot(sim_occs_list, data_1, col_1, label_1, title_col_1, plt_title)
    
    # Filter data for relevant occupations for 2023
    occs = set([title[0] for title in sim_occs_list])
    subset_data = data_2[data_2[title_col_2].isin(occs)].drop_duplicates(subset=[title_col_2], keep='first')
    
    # Extract data for 2023
    col_data_2 = list(subset_data[col_2])
    subset_titles = list(subset_data[title_col_2])
    
    # Add 2023 data to the figure
    fig.add_trace(go.Bar(
        x=subset_titles,
        y=col_data_2,
        name=label_2,
        marker_color='salmon'
    ))
    
    # Update layout for grouped bars
    fig.update_layout(
        barmode='group'  # Grouped bar chart
    )
    
    return fig

def create_gauge_chart(zipped_list, occ_edu_data):
    fig = go.Figure()

    # Loop through each item in zipped_list and create a gauge for each
    for i in range(len(zipped_list)):
        fig.add_trace(go.Indicator(
            mode="number+gauge", 
            value=100 - zipped_list[i][1],
            domain={'x': [0.25, 1], 'y': [0.2 * i, 0.2 * i + 0.15]},  # Adjust the y-domain for each gauge
            title={
                'text': f'{zipped_list[i][0][:35]}<br>{zipped_list[i][0][35:]}',
                'font': {'size': 14}
            },
            number={'font': {'size': 20}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, sum(list(occ_edu_data.iloc[i][2:4]))], 'color': "darkred"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:3])), sum(list(occ_edu_data.iloc[i][2:4]))], 'color': "red"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:4])), sum(list(occ_edu_data.iloc[i][2:5]))], 'color': "orange"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:5])), sum(list(occ_edu_data.iloc[i][2:6]))], 'color': "yellow"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:6])), sum(list(occ_edu_data.iloc[i][2:7]))], 'color': "lightgreen"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:7])), sum(list(occ_edu_data.iloc[i][2:8]))], 'color': "green"},
                    {'range': [sum(list(occ_edu_data.iloc[i][2:8])), sum(list(occ_edu_data.iloc[i][2:9]))], 'color': "darkgreen"}
                ],
                'bar': {'color': "black"}
            }
        ))

    fig.update_layout(
        height=800,  # Adjust height to accommodate all gauges
        width=1000,  # Adjust width
        title="Occupation Education Level Gauge"
    )

    return fig

#col_name = '2023 National Employment Matrix title'
def get_user_education_percentile(user_education, sim_occs_list, educ_data, col_name, tokenizer, model):
    occs = set([title[0] for title in sim_occs_list])
    occs = [x.lower() for x in occs]

    educ_data[col_name] = educ_data[col_name].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    subset_educ_data = educ_data[educ_data[col_name].isin(occs)].drop_duplicates(subset = [col_name], keep = 'first')
    
    eq_degree = get_similar_items(user_education, list(educ_data.columns), tokenizer, model)

    most_similar_index = eq_degree[0][0]
    user_pct_rank = []
    
    for i in range(0, 5):
        # Extract the relevant data from the row
        pct_list = subset_educ_data.iloc[i][most_similar_index:]
        proc_pct_list = []
        for value in pct_list:
            try:
                proc_pct_list.append(int(value))
            except ValueError:
                proc_pct_list.append(0)
        user_pct_rank.append(sum(proc_pct_list))
        
    zipped_list = list(zip(subset_educ_data[col_name][:5], user_pct_rank))
    figure = create_gauge_chart(zipped_list, subset_educ_data)
    return figure


def create_projection_chart(sim_occs):
    # Extract the required data from the input DataFrame (sim_occs)
    x = list(sim_occs.iloc[:5, 0])  # Occupation titles
    percent_change_2019_29 = list(sim_occs.iloc[:5, 6])  # Percent change projections 2019-29
    percent_change_2023_33 = list(sim_occs.iloc[:5, 19])  # Percent change projections 2023-33
    annual_openenings_19_29 = list(sim_occs.iloc[:5, 13])  # Annual average openings 2019-29
    annual_openenings_23_33 = list(sim_occs.iloc[:5, 26])  # Annual average openings 2023-33

    # Create the Plotly figure with initial data (percent change)
    fig = go.Figure()

    # Add traces for percent change (default view)
    fig.add_trace(go.Bar(
        x=x,
        y=percent_change_2019_29,
        name='Percent Change Projections 2019-29',
        marker_color='skyblue'
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=percent_change_2023_33,
        name='Percent Change Projections 2023-33',
        marker_color='salmon'
    ))

    # Add traces for annual openings (initially hidden)
    fig.add_trace(go.Bar(
        x=x,
        y=annual_openenings_19_29,
        name='Annual Openings Projections 2019-29',
        marker_color='lightgreen',
        visible=False
    ))
    fig.add_trace(go.Bar(
        x=x,
        y=annual_openenings_23_33,
        name='Annual Openings Projections 2023-33',
        marker_color='orange',
        visible=False
    ))

    # Update layout with dropdown menu
    fig.update_layout(
        title={
            'text': '2019 vs 2023 Projections for Total Employment Change Percentage',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Occupation Titles',
        yaxis_title='Percent Change',
        barmode='group',  # Group bars side by side
        height=800,
        width=1150,
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label='Percent Change',
                        method='update',
                        args=[
                            {'visible': [True, True, False, False]},  # Show percent change bars
                            {
                                'title': '2019 vs 2023 Projections for Total Employment Change Percentage',
                                'yaxis': {'title': 'Percent Change'}
                            }
                        ]
                    ),
                    dict(
                        label='Annual Openings',
                        method='update',
                        args=[
                            {'visible': [False, False, True, True]},  # Show annual openings bars
                            {
                                'title': '2019 vs 2023 Projections for Annual Average Openings',
                                'yaxis': {'title': 'Annual Openings'}
                            }
                        ]
                    )
                ],
                direction='down',  # Dropdown menu direction
                showactive=True,  # Highlight the selected option
                x=0.1,
                y=1.2
            )
        ]
    )

    return fig



# chg_proj_2019 = pd.read_excel('project_data/2019-29/occupation.xlsx', sheet_name = 8, skiprows = 1)
# chg_proj_2023 = pd.read_excel('project_data/2023-33/occupation.xlsx', sheet_name = 10, skiprows = 1)
#user_occ = profile['industryName']


def get_comparitive_projections(chg_proj_2019, chg_proj_2023, sim_occs_list):
    chg_proj_2019['2019 National Employment Matrix title'] = chg_proj_2019['2019 National Employment Matrix title'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower().str.strip()
    chg_proj_2023['2023 National Employment Matrix title'] = chg_proj_2023['2023 National Employment Matrix title'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower().str.strip()
    chg_proj_2019.rename(columns={'2019 National Employment Matrix code':'Matrix code'}, inplace = True)
    chg_proj_2023.rename(columns={'2023 National Employment Matrix code':'Matrix code'}, inplace = True)
    
    chg_proj_merge = pd.merge(chg_proj_2019, chg_proj_2023, how='inner', on = 'Matrix code')
    
    occs = set([title[0] for title in sim_occs_list])
    occs = [x.lower() for x in occs]
    
    sim_occs = chg_proj_merge[chg_proj_merge['2023 National Employment Matrix title'].isin(occs)].drop_duplicates(subset = ['2023 National Employment Matrix title'], keep = 'first')
    figure = create_projection_chart(sim_occs)
    return figure




    

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from threading import Thread
import os 
import time



# Create Dash application
app = dash.Dash(__name__)

# Define the app layout
app.layout = html.Div([
    html.H1("Career Projection Dashboard"),

    # Input box for LinkedIn handle
    html.Label("Enter your LinkedIn handle:"),
    dcc.Input(id='linkedin-handle', type='text', placeholder='LinkedIn Handle'),

    # Display the four plots
    html.Div([
        html.Div([
            dcc.Graph(id='figure-1'),
        ], className='six columns'),

        html.Div([
            dcc.Graph(id='figure-2'),
        ], className='six columns'),

    ], className='row'),

    html.Div([
        html.Div([
            dcc.Graph(id='figure-3'),
        ], className='six columns'),

        html.Div([
            dcc.Graph(id='figure-4'),
        ], className='six columns'),

    ], className='row'),
])

# Define the callback to update graphs
@app.callback(
    [Output('figure-1', 'figure'),
     Output('figure-2', 'figure'),
     Output('figure-3', 'figure'),
     Output('figure-4', 'figure')],
    [Input('linkedin-handle', 'value')]
)
def update_figures(linkedin_handle):
    profile = get_linkedin_profile('gze.pois0n@gmail.com', 'password',linkedin_handle)
    #'rupobrata-panja-554b6b102'
    data_2023, data_2019, ind_data_2033, educ_data_2023, chg_proj_2019, chg_proj_2023 = initialize_dataframes()
    tokenizer,model = init_model()
    sim_occs_list = get_similar_items(profile['industryName'], data_2023['OCC_TITLE'], tokenizer, model)
    figure_1 = single_bar_plot(
                    sim_occs_list=sim_occs_list, 
                    data=data_2019, 
                    col_1='h_mean', 
                    label_1='Hourly Salary ($)', 
                    title_col = 'occ_title',
                    plt_title='Hourly Salaries of Top Similar Occupations'
                        )
    print('figure 1 done')
    figure_2 = double_bar_plot(
                    sim_occs_list=sim_occs_list,
                    data_1=data_2019,
                    data_2=data_2023,
                    col_1='h_mean',
                    col_2='H_MEAN',
                    label_1='Hourly Salary ($) 2020',
                    label_2='Hourly Salary ($) 2023',
                    title_col_1 = 'occ_title',
                    title_col_2 = 'OCC_TITLE',
                    plt_title='Hourly Salaries Comparison: 2020 vs 2023'
                )
    user_education = profile['education'][0]['degreeName']
    print('figure 2 done')
    user_education = re.sub(r'[^a-zA-Z\s]', '', user_education).lower()
    figure_3 = get_user_education_percentile(user_education = user_education, 
                                                sim_occs_list = sim_occs_list, 
                                                educ_data = educ_data_2023, 
                                                col_name = '2023 National Employment Matrix title', 
                                                tokenizer = tokenizer, model = model )
    print('figure 3 done')
    figure_4 = get_comparitive_projections(chg_proj_2019, chg_proj_2023, sim_occs_list)
    print("calculations done")
    return figure_1, figure_2, figure_3, figure_4
    
def kill_app():
    time.sleep(25
               )  # Sleep for 3 minutes (180 seconds)
    print("App is being closed after 3 minutes.")
    os._exit(0)  # Terminate the app

# Start the kill_app function in a separate thread
Thread(target=kill_app).start()

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)