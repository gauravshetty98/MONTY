from LinkedinProfileFetcher import LinkedInProfileFetcher
from DataAndModelInitializer import DataAndModelInitializer
from EmbeddingProcessor import EmbeddingProcessor
from VisualizationTools import VisualizationTools

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from threading import Thread
import os 
import time
import re


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
    # Initialize LinkedIn profile object
    profile_object = LinkedInProfileFetcher('gze.pois0n@gmail.com', 'password')

    # Fetch LinkedIn profile data
    profile = profile_object.get_profile(profile_id= linkedin_handle)
    
    #'rupobrata-panja-554b6b102'

     # Initialize dataframes and model
    data_initializer = DataAndModelInitializer()
    dataframes = data_initializer.initialize_dataframes()
    tokenizer, model = data_initializer.init_model()

    #data_2023, data_2019, ind_data_2033, educ_data_2023, chg_proj_2019, chg_proj_2023 = initialize_dataframes()
    
    # Initialize embedding processor
    embedding_processor = EmbeddingProcessor(tokenizer, model)

    # Load or compute embeddings for occupation titles
    occ_title_embeddings = embedding_processor.load_embeddings(dataframes['data_2023']['OCC_TITLE'], batch_size=64)

    # Get top 10 similar occupations to LinkedIn profile's industry
    sim_occs_list = embedding_processor.get_similar_items(profile['industryName'], occ_title_embeddings)

    vtool = VisualizationTools(tokenizer, model)

    figure_1 = vtool.single_bar_plot(
                    sim_occs_list=sim_occs_list, 
                    data=dataframes['data_2023'], 
                    col_1='H_MEAN', 
                    label_1='Hourly Salary ($)', 
                    title_col = 'OCC_TITLE',
                    plt_title='Hourly Salaries of Top Similar Occupations'
                        )
    print('figure 1 done')


    figure_2 = vtool.double_bar_plot(
                    sim_occs_list=sim_occs_list,
                    data_1=dataframes['data_2019'],
                    data_2=dataframes['data_2023'],
                    col_1='h_mean',
                    col_2='H_MEAN',
                    label_1='Hourly Salary ($) 2020',
                    label_2='Hourly Salary ($) 2023',
                    title_col_1 = 'occ_title',
                    title_col_2 = 'OCC_TITLE',
                    plt_title='Hourly Salaries Comparison: 2020 vs 2023'
                )
    print('figure 2 done')
    
    user_education = profile['education'][0]['degreeName']
    user_education = re.sub(r'[^a-zA-Z\s]', '', user_education).lower()
    figure_3 = vtool.get_user_education_percentile(user_education = user_education, 
                                                sim_occs_list = sim_occs_list, 
                                                educ_data = dataframes['educ_data_2023'], 
                                                col_name = '2023 National Employment Matrix title', 
                                                tokenizer = tokenizer, model = model )
    print('figure 3 done')


    figure_4 = vtool.get_comparative_projections(dataframes['chg_proj_2019'], dataframes['chg_proj_2023'], sim_occs_list)
    print("calculations done")
    return figure_1, figure_2, figure_3, figure_4
    
def kill_app():
    time.sleep(120)  # Sleep for 3 minutes (180 seconds)
    print("App is being closed after 3 minutes.")
    os._exit(0)  # Terminate the app

# Start the kill_app function in a separate thread
Thread(target=kill_app).start()

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)