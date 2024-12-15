from LinkedinProfileFetcher import LinkedInProfileFetcher
from DataAndModelInitializer import DataAndModelInitializer
from EmbeddingProcessor import EmbeddingProcessor
from VisualizationTools import VisualizationTools
from MapCreator import MapCreator

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from threading import Thread
import os 
import time
import re

# Create Dash application
app = dash.Dash(__name__)

# Define the app layout
# Define the app layout
app.layout = html.Div(
    style={
        'backgroundColor': '#f4f6f9',
        'fontFamily': 'Arial, sans-serif',
        'padding': '20px',
        'overflowX': 'auto'  # Ensure horizontal scrolling when needed
    },
    children=[
       # Header and Input Section
        html.Div([
            html.H1("Industry Overview Dashboard",
                    style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'color': '#3c3c3c'}),
            html.Label("Enter your LinkedIn handle:",
                       style={'fontSize': '18px', 'fontFamily': 'Arial, sans-serif', 'color': '#555'}),
            html.Div([
                dcc.Input(id='linkedin-handle', type='text', placeholder='LinkedIn Handle',
                          style={'width': '50%', 'padding': '20px', 'fontSize': '16px', 'borderRadius': '5px'}),
                html.Button('Analyze', id='analyze-button', n_clicks=0,
                            style={'marginLeft': '10px', 'padding': '10px 20px', 'fontSize': '16px',
                                   'border': 'none', 'borderRadius': '5px', 'backgroundColor': '#007BFF',
                                   'color': 'white', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'margin': '20px 0'}),
        ], style={'padding': '30px', 'backgroundColor': '#f4f4f9'}),

       # User Industry Information Section
        html.Div([
            # html.Label("Industry Information:",
            #            style={'fontSize': '18px', 'fontFamily': 'Arial, sans-serif', 'color': '#555'}),
            dcc.Textarea(id='user-industry', value='', readOnly=True,
                         style={'width': '100%', 'height': '60px', 'padding': '10px', 'fontSize': '16px',
                                'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),
            html.Div([
                html.Button('Continue', id='continue-button', n_clicks=0,
                            style={'marginTop': '10px', 'padding': '10px 20px', 'fontSize': '16px',
                                   'border': 'none', 'borderRadius': '5px', 'backgroundColor': '#28A745',
                                   'color': 'white', 'cursor': 'pointer'}),
                html.Button('Change Occupation', id='change-occupation-button', n_clicks=0,
                            style={'marginTop': '10px', 'marginLeft': '10px', 'padding': '10px 20px', 'fontSize': '16px',
                                   'border': 'none', 'borderRadius': '5px', 'backgroundColor': '#FFC107',
                                   'color': 'white', 'cursor': 'pointer'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={'margin': '20px 0', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'start'}),

        # Dialog Box for Changing Industry
        html.Div(id='change-industry-dialog', style={
            'display': 'none', 
            'marginTop': '20px', 
            'padding': '20px', 
            'border': '1px solid #ddd', 
            'borderRadius': '5px', 
            'backgroundColor': '#f9f9f9'
        }, children=[
            html.Label("Enter Industry Name:",
                    style={'fontSize': '18px', 'fontFamily': 'Arial, sans-serif', 'color': '#555'}),
            dcc.Input(id='custom-industry', type='text', placeholder='Enter custom industry name',
                    style={'width': '100%', 'padding': '10px', 'fontSize': '16px', 'borderRadius': '5px'}),
            html.Button('Submit', id='submit-industry', n_clicks=0,
                        style={'marginTop': '10px', 'padding': '10px 20px', 'fontSize': '16px',
                            'border': 'none', 'borderRadius': '5px', 'backgroundColor': '#007BFF',
                            'color': 'white', 'cursor': 'pointer'}),
        ]),

        # Analysis Section (Initially Hidden)
        # Analysis Section (Updated with Text Boxes)
html.Div(id='analysis-section', style={'display': 'none'}, children=[
    # Text and Graph Row 1
    html.Div([
        html.Div([
            html.P("These are the annual salaries of some occupation based on the industry information. This can be used to get an idea about different occupations and compare their compensation",
                   style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Loading(
                id='loading-figure-1',
                type='circle',
                children=dcc.Graph(id='figure-1', config={'displayModeBar': False},
                                   style={ 'borderRadius': '5px'}),
            ),
        ], className='six columns', style={'padding': '20px'}),

        html.Div([
            html.P("Here we try to analyze the annual salaries of those occupations pre-covid (2019) and post-covid(2013).",
                   style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Loading(
                id='loading-figure-2',
                type='circle',
                children=dcc.Graph(id='figure-2', config={'displayModeBar': False},
                                   style={ 'borderRadius': '5px'}),
            ),
        ], className='six columns', style={'padding': '20px'}),
    ], className='row', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around', 'padding': '20px'}),

    # Text and Graph Row 2
    html.Div([
        html.Div([
            html.P("We also want to show how the workers for each occupation are distributed based on their educational qualifications. This will help get an idea where you stand based on the education qualification",
                   style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Loading(
                id='loading-figure-3',
                type='circle',
                children=dcc.Graph(id='figure-3', config={'displayModeBar': False},
                                   style={ 'borderRadius': '5px'}),
            ),
        ], className='six columns', style={'padding': '20px'}),

        html.Div([
            html.P("Here we try to analyze how the projections and openings have changed pre-covid and post-covid. Whether Covid has lead to decrease or increase in employement.",
                   style={'marginBottom': '10px', 'fontSize': '16px'}),
            dcc.Loading(
                id='loading-figure-4',
                type='circle',
                children=dcc.Graph(id='figure-4', config={'displayModeBar': False},
                                   style={ 'borderRadius': '5px'}),
            ),
        ], className='six columns', style={'padding': '20px'}),
    ], className='row', style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'padding': '20px'}),

    # Text and Figure 5 with Spacing
    html.Div([
        dcc.Loading(
            id='loading-figure-5',
            type='circle',
            children=dcc.Graph(id='figure-5', config={'displayModeBar': False},
                               style={'borderRadius': '5px'}),
        ),
    ], className='row', style={'padding': '20px', 'display': 'flex', 'justifyContent': 'center'}),

    html.Div([
    html.Label("Select Educational Qualification:", style={'fontSize': '18px', 'fontFamily': 'Arial, sans-serif', 'color': '#555'}),
    dcc.RadioItems(
        id='degree-selector',
        options=[
            {'label': "No formal educational credential", 'value': "No formal educational credential"},
            {'label': "High school diploma or equivalent", 'value': "High school diploma or equivalent"},
            {'label': "Some college, no degree", 'value': "Some college, no degree"},
            {'label': "Postsecondary nondegree award", 'value': "Postsecondary nondegree award"},
            {'label': "Associate's degree", 'value': "Associate's degree"},
            {'label': "Bachelor's degree", 'value': "Bachelor's degree"},
            {'label': "Master's degree", 'value': "Master's degree"},
            {'label': "Doctoral or professional degree", 'value': "Doctoral or professional degree"}
        ],
        value="Master's degree",  # Default value
        style={'marginTop': '10px', 'fontSize': '16px'},
        inputStyle={'marginRight': '10px'}
    ),
], style={'padding': '20px', 'backgroundColor': '#f4f6f9'}),

    # Text and Map Frame 1
    html.Div([
        html.P("This is a map created using Folium and BLS data. We have also implemented GeoPy to get lat-longs of different areas mentioned in BLS. This will give you an idea of how the salaries are distributed for those occupations, based on the location and educational qualifications.",
               style={'marginBottom': '10px', 'fontSize': '16px'}),
        dcc.Loading(
            id='loading-map-frame-1',
            type='circle',
            children=html.Iframe(
                id='map-frame-1',
                srcDoc=open('a_mean_wrt_degree_map.html', 'r').read(),  # Will be updated dynamically
                width='100%',
                height='600',
                style={'border': '1px solid #ddd', 'borderRadius': '5px'}
            ),
        ),
    ], className='row', style={'marginTop': '20px', 'padding': '20px'}),

    # Text and Map Frame 2
    html.Div([
        html.P("This is a map created using Folium and BLS data. We have also implemented GeoPy to get lat-longs of different areas mentioned in BLS.  This will give you an idea of how the total employement is distributed for those occupations, based on the location and educational qualifications.",
               style={'marginBottom': '10px', 'fontSize': '16px'}),
        dcc.Loading(
            id='loading-map-frame-2',
            type='circle',
            children=html.Iframe(
                id='map-frame-2',
                srcDoc=open('tot_emp_wrt_degree_map.html', 'r').read(),  # Will be updated dynamically
                width='100%',
                height='600',
                style={'border': '1px solid #ddd', 'borderRadius': '5px'}
            ),
        ),
    ], className='row', style={'marginTop': '20px', 'padding': '20px'}),
]),

    ]
)


# Callback to fetch user's industry
@app.callback(
    Output('user-industry', 'value'),
    Input('analyze-button', 'n_clicks'),
    State('linkedin-handle', 'value')
)
def fetch_user_industry(n_clicks, linkedin_handle):
    if n_clicks == 0:
        return ''
    profile_object = LinkedInProfileFetcher('gze.pois0n@gmail.com', 'password')
    profile = profile_object.get_profile(profile_id=linkedin_handle)
    user_industry = profile.get('industryName', 'Unknown Industry')
    return f"We have fetched your profile from LinkedIn, and based on the fetched information, it seems that you work in the '{user_industry}' industry. We have created some analysis based on your profile, education and experiences. If this is not the industry of your choice, you can enter any other industry below"



# Callback to show dialog for changing industry
@app.callback(
    Output('change-industry-dialog', 'style'),
    Input('change-occupation-button', 'n_clicks')
)
def show_change_industry_dialog(n_clicks):
    if n_clicks > 0:
        return {'display': 'block'}
    return {'display': 'none'}


# Callback to show analysis section
@app.callback(
    Output('analysis-section', 'style'),
    Input('continue-button', 'n_clicks')
)
def show_analysis_section(n_clicks):
    if n_clicks > 0:
        return {'display': 'block'}
    return {'display': 'none'}

@app.callback(
    [Output('map-frame-1', 'srcDoc', allow_duplicate=True),
     Output('map-frame-2', 'srcDoc', allow_duplicate=True)],
    Input('degree-selector', 'value'),
    prevent_initial_call=True
)
def update_maps(degree_name):
    # Initialize the MapCreator class with the path to the latitude-longitude data
    map_creator = MapCreator(lat_lon_path='project_data/extra/bls_area_lat_long.xlsx')

    # Load the dataframes and years
    dataframes, years = map_creator.load_data()

    # Create map for annual mean based on degree and location
    final_df = map_creator.create_combined_df(dataframes, years, column_name="a_mean")
    map_vis = map_creator.heatmap_with_time(final_df, degree_name=degree_name, column_name="a_mean")
    a_mean_map_file_path = 'a_mean_wrt_degree_map.html'
    map_vis.save(a_mean_map_file_path)
    with open(a_mean_map_file_path, 'r') as f:
        a_mean_map_html_content = f.read()

    # Create map for total employment based on degree and location
    final_df = map_creator.create_combined_df(dataframes, years, column_name="tot_emp")
    map_vis = map_creator.heatmap_with_time(final_df, degree_name=degree_name, column_name="tot_emp")
    tot_emp_map_file_path = 'tot_emp_wrt_degree_map.html'
    map_vis.save(tot_emp_map_file_path)
    with open(tot_emp_map_file_path, 'r') as f:
        tot_emp_map_html_content = f.read()

    return a_mean_map_html_content, tot_emp_map_html_content


@app.callback(
    [Output('figure-1', 'figure'),
     Output('figure-2', 'figure'),
     Output('figure-3', 'figure'),
     Output('figure-4', 'figure'),
     Output('figure-5', 'figure'),
     Output('map-frame-1', 'srcDoc'),
     Output('map-frame-2', 'srcDoc')],
    [Input('submit-industry', 'n_clicks'),
     Input('continue-button', 'n_clicks')],
    [State('custom-industry', 'value'),
     State('linkedin-handle', 'value')]
)
def update_figures(submit_clicks, continue_clicks, custom_industry, linkedin_handle):
  # Added degree_name
    profile_object = LinkedInProfileFetcher('gze.pois0n@gmail.com', 'password')
    profile = profile_object.get_profile(profile_id=linkedin_handle)
    
    if submit_clicks > 0 and custom_industry:
        user_industry = custom_industry
    else:
        # Constructing the user_industry description
        profile_ed = ''
        profile_exp = ''
        if 'education' in profile and isinstance(profile['education'], list):
            for edu in profile['education']:
                field = edu.get('fieldOfStudy', '')
                if field:
                    profile_ed += field + ' '
        if 'experience' in profile and isinstance(profile['experience'], list):
            for exp in profile['experience']:
                title = exp.get('title', '')
                if title:
                    profile_exp += title + ' '
        user_industry = f"{profile.get('industryName')} {profile_exp.strip()} {profile_ed.strip()}"

    print(user_industry)
    data_initializer = DataAndModelInitializer()
    dataframes = data_initializer.initialize_dataframes()
    tokenizer, model = data_initializer.init_model()
    embedding_processor = EmbeddingProcessor(tokenizer, model)

    occ_title_embeddings = embedding_processor.load_embeddings(dataframes['data_2023']['OCC_TITLE'], batch_size=64)
    sim_occs_list = embedding_processor.get_similar_items(user_industry, occ_title_embeddings)
    print(sim_occs_list)

    vtool = VisualizationTools(tokenizer, model)

    vtool = VisualizationTools(tokenizer, model)

    figure_1 = vtool.single_bar_plot(
                    sim_occs_list=sim_occs_list, 
                    data=dataframes['data_2023'], 
                    col_1='A_MEAN', 
                    label_1='Annual Salary ($)', 
                    title_col = 'OCC_TITLE',
                    plt_title='Annual Salary Of Similar Occupations'
                        )

    figure_2 = vtool.double_bar_plot(
                    sim_occs_list=sim_occs_list,
                    data_1=dataframes['data_2019'],
                    data_2=dataframes['data_2023'],
                    col_1='a_mean',
                    col_2='A_MEAN',
                    label_1='Annual Salary ($) 2019',
                    label_2='Annual Salary ($) 2023',
                    title_col_1 = 'occ_title',
                    title_col_2 = 'OCC_TITLE',
                    plt_title='Annual Salary Comparison: 2019 vs 2023'
                )

    # user_education = profile['education'][0]['degreeName']
    # user_education = re.sub(r'[^a-zA-Z\s]', '', user_education).lower()
    # figure_3 = vtool.get_user_education_percentile(user_education = user_education, 
    #                                             sim_occs_list = sim_occs_list, 
    #                                             educ_data = dataframes['educ_data_2023'], 
    #                                             col_name = '2023 National Employment Matrix title', 
    #                                             tokenizer = tokenizer, model = model )
    
    figure_3 = vtool.create_heatmap(educ_data=dataframes['educ_data_2023'], 
                                    occupations=[title[0].lower() for title in sim_occs_list], 
                                    col_name='2023 National Employment Matrix title'
                                    )

    figure_4 = vtool.get_comparative_projections(dataframes['chg_proj_2019'], dataframes['chg_proj_2023'], sim_occs_list)

    education_mapping = {
                        'No formal educational credential': 1,
                        'Some college, no degree': 1,
                        'Postsecondary nondegree award': 3,
                        'High school diploma or equivalent': 2,
                        "Associate's degree": 4,
                        "Bachelor's degree": 5,
                        "Master's degree": 6,
                        'Doctoral or professional degree': 7
                    }
    
    figure_5 = vtool.double_bar_plot(
                                    sim_occs_list=sim_occs_list,
                                    data_1=dataframes['occ_ed_req_19'],
                                    data_2=dataframes['occ_ed_req_23'],
                                    col_1='Education Level Numeric',
                                    col_2='Education Level Numeric',
                                    label_1='Minimum Education Requirements 2019',
                                    label_2='Minimum Education Requirements 2023',
                                    title_col_1 = 'OES May 2019 Title',
                                    title_col_2 = 'OEWS May 2023 Title',
                                    plt_title='Minimum Educational Requirements: 2019 vs 2023',
                                    education_mapping=education_mapping
                                    )

    # Initialize the MapCreator class with the path to the latitude-longitude data
    map_creator = MapCreator(lat_lon_path='project_data/extra/bls_area_lat_long.xlsx')

    # Load the dataframes and years
    dataframes, years = map_creator.load_data()

    # Create map for annual mean based on degree and location
    final_df = map_creator.create_combined_df(dataframes, years, column_name = "a_mean")
    map_vis = map_creator.heatmap_with_time(final_df, degree_name = "Master's degree", column_name = "a_mean", normalize=False)
    a_mean_map_file_path = 'a_mean_wrt_degree_map.html'
    map_vis.save(a_mean_map_file_path)
    with open(a_mean_map_file_path, 'r') as f:
        a_mean_map_html_content = f.read()

    # Create map for total employment based on degree and location
    final_df = map_creator.create_combined_df(dataframes, years, column_name = "tot_emp")
    map_vis = map_creator.heatmap_with_time(final_df, degree_name = "Master's degree", column_name = "tot_emp", normalize=False)
    tot_emp_map_file_path = 'tot_mean_wrt_degree_map.html'
    map_vis.save(tot_emp_map_file_path)
    with open(tot_emp_map_file_path, 'r') as f:
        tot_emp_map_html_content = f.read()

    print("done")

    return figure_1, figure_2, figure_3, figure_4, figure_5, a_mean_map_html_content, tot_emp_map_html_content


# def kill_app():
#     time.sleep(180)  # Sleep for 3 minutes (180 seconds)
#     print("App is being closed after 3 minutes.")
#     os._exit(0)  # Terminate the app

# Start the kill_app function in a separate threadç
#Thread(target=kill_app).start()

# Run the server
if __name__ == '__main__':
    app.run_server(debug=False)
    #update_figures(1, 1, None, 'josephmathew0')