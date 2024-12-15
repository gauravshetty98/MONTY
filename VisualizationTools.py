import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModel
from EmbeddingProcessor import EmbeddingProcessor

class VisualizationTools:
    def __init__(self, tokenizer=None, model=None):
        """
        Initialize the VisualizationTools class with optional tokenizer and model for text processing.
        
        Args:
            tokenizer: Pretrained tokenizer for text processing (default: None).
            model: Pretrained model for embeddings or similarity calculations (default: None).
        """
        self.tokenizer = tokenizer
        self.model = model

    def single_bar_plot(self, sim_occs_list, data, col_1, label_1, title_col, plt_title):
        # Extract unique titles in the same order as they appear in sim_occs_list
        occs = [title[0] for title in sim_occs_list]
        occs = list(dict.fromkeys(occs))  # Remove duplicates while preserving order

        # Filter data based on ordered titles
        salary_data = data[data[title_col].isin(occs)].drop_duplicates(subset=[title_col], keep='first')

        # Ensure only valid occupations are used for reordering
        valid_occs = [occ for occ in occs if occ in salary_data[title_col].values]
        if not valid_occs:
            print("No matching occupations found in the data.")
            return go.Figure()

        # Reorder salary_data to match the order in valid_occs
        salary_data = salary_data.set_index(title_col).loc[valid_occs].reset_index()

        # Extract data for plotting
        hourly_sals = list(salary_data[col_1])
        occ_titles = list(salary_data[title_col])

        # Create the bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=occ_titles,
            y=hourly_sals,
            name=label_1,
            marker_color='skyblue'
        ))

        # Update layout
        fig.update_layout(
            title=plt_title,
            xaxis=dict(title='Occupation Titles', tickangle=45, automargin=True),
            yaxis=dict(title=label_1, titlefont=dict(color='skyblue'), tickfont=dict(color='skyblue')),
            height=600,
            width=1000
        )
        return fig


    def double_bar_plot(self, sim_occs_list, data_1, data_2, col_1, col_2, label_1, label_2, title_col_1, title_col_2, plt_title, education_mapping=None):
        """
        Create a bar chart comparing two datasets by adding a second bar trace
        to the figure returned by single_bar_plot.

        Parameters:
            sim_occs_list (list): List of similar occupations.
            data_1 (DataFrame): Data for the first dataset.
            data_2 (DataFrame): Data for the second dataset.
            col_1 (str): Column name for the first dataset.
            col_2 (str): Column name for the second dataset.
            label_1 (str): Label for the first dataset.
            label_2 (str): Label for the second dataset.
            title_col_1 (str): Column name for titles in the first dataset.
            title_col_2 (str): Column name for titles in the second dataset.
            plt_title (str): Title of the plot.
            education_mapping (dict or None): Mapping for education levels (if applicable).
            
        Returns:
            plotly.graph_objects.Figure: The generated figure.
        """
        # Get the single bar plot for the first dataset
        fig = self.single_bar_plot(sim_occs_list, data_1, col_1, label_1, title_col_1, plt_title)
        
        # Filter data for relevant occupations for the second dataset
        occs = set([title[0] for title in sim_occs_list])
        subset_data = data_2[data_2[title_col_2].isin(occs)].drop_duplicates(subset=[title_col_2], keep='first')
        
        # Extract data for the second dataset
        col_data_2 = list(subset_data[col_2])
        subset_titles = list(subset_data[title_col_2])
        
        # Add second dataset data to the figure
        fig.add_trace(go.Bar(
            x=subset_titles,
            y=col_data_2,
            name=label_2,
            marker_color='salmon'
        ))
        
        # Update layout for grouped bars
        fig.update_layout(
            barmode='group',  # Grouped bar chart
            title=plt_title
        )
        
        # Apply education level mapping to y-axis if provided
        if education_mapping:
            fig.update_yaxes(
                tickvals=list(education_mapping.values()),
                ticktext=list(education_mapping.keys())
            )
        
        return fig


    def create_gauge_chart(self, zipped_list, occ_edu_data):
        fig = go.Figure()

        for i, (title, rank) in enumerate(zipped_list):
            fig.add_trace(go.Indicator(
                mode="number+gauge",
                value=100 - rank,
                domain={'x': [0.25, 1], 'y': [0.2 * i, 0.2 * i + 0.15]},
                title={'text': f'{title[:35]}<br>{title[35:]}', 'font': {'size': 14}},
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
            height=800,
            width=1000,
            title="Occupation Education Level Gauge"
        )
        return fig
    
    def get_user_education_percentile(self, user_education, sim_occs_list, educ_data, col_name, tokenizer, model):
        # Extract unique occupation titles in the order they appear in sim_occs_list
        occs = [title[0].lower() for title in sim_occs_list]
        occs = list(dict.fromkeys(occs))  # Remove duplicates while preserving order

        # Clean and preprocess the education data column
        educ_data[col_name] = educ_data[col_name].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

        # Filter and reorder subset_educ_data based on the order of valid_occs
        subset_educ_data = educ_data[educ_data[col_name].isin(occs)].drop_duplicates(subset=[col_name], keep='first')
        valid_occs = [occ for occ in occs if occ in subset_educ_data[col_name].values]
        subset_educ_data = subset_educ_data.set_index(col_name).loc[valid_occs].reset_index()

        # Initialize the embedding processor and calculate similarity
        ep = EmbeddingProcessor(tokenizer, model)
        embeddings = ep.load_embeddings(list(educ_data.columns), 32, filename="edu_embeddings.npy")
        eq_degree = ep.get_similar_items(user_education, embeddings)

        most_similar_index = eq_degree[0][0]
        user_pct_rank = []
        

        #print(subset_educ_data)

        # Extract percentile data for the top 5 similar occupations
        for i in range(0, 5):
            pct_list = subset_educ_data.iloc[i][most_similar_index:]
            proc_pct_list = []
            for value in pct_list:
                try:
                    proc_pct_list.append(int(value))
                except ValueError:
                    proc_pct_list.append(0)
            user_pct_rank.append(sum(proc_pct_list))

        # Combine data for the gauge chart
        zipped_list = list(zip(subset_educ_data[col_name][:5], user_pct_rank))
        figure = self.create_gauge_chart(zipped_list, subset_educ_data)
        return figure





    def create_projection_chart(self, sim_occs):
        x = list(sim_occs.iloc[:5, 0])
        percent_change_2019_29 = list(sim_occs.iloc[:5, 6])
        percent_change_2023_33 = list(sim_occs.iloc[:5, 19])
        annual_openenings_19_29 = list(sim_occs.iloc[:5, 13])
        annual_openenings_23_33 = list(sim_occs.iloc[:5, 26])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=percent_change_2019_29, name='Percent Change Projections 2019-29', marker_color='skyblue'))
        fig.add_trace(go.Bar(x=x, y=percent_change_2023_33, name='Percent Change Projections 2023-33', marker_color='salmon'))
        fig.add_trace(go.Bar(x=x, y=annual_openenings_19_29, name='Annual Openings Projections 2019-29', marker_color='lightgreen', visible=False))
        fig.add_trace(go.Bar(x=x, y=annual_openenings_23_33, name='Annual Openings Projections 2023-33', marker_color='orange', visible=False))

        fig.update_layout(
            title={'text': '2019 vs 2023 Projections for Total Employment Change Percentage', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Occupation Titles',
            yaxis_title='Percent Change',
            barmode='group',
            height=800,
            width=1150,
            updatemenus=[
                dict(
                    buttons=[
                        dict(label='Percent Change', method='update', args=[{'visible': [True, True, False, False]}, {'yaxis': {'title': 'Percent Change'}}]),
                        dict(label='Annual Openings', method='update', args=[{'visible': [False, False, True, True]}, {'yaxis': {'title': 'Annual Openings'}}])
                    ],
                    direction='down',
                    showactive=True,
                    x=0.1,
                    y=1.2
                )
            ]
        )
        return fig

    def get_comparative_projections(self, chg_proj_2019, chg_proj_2023, sim_occs_list):
        chg_proj_2019['2019 National Employment Matrix title'] = chg_proj_2019['2019 National Employment Matrix title'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower().str.strip()
        chg_proj_2023['2023 National Employment Matrix title'] = chg_proj_2023['2023 National Employment Matrix title'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower().str.strip()
        chg_proj_2019.rename(columns={'2019 National Employment Matrix code': 'Matrix code'}, inplace=True)
        chg_proj_2023.rename(columns={'2023 National Employment Matrix code': 'Matrix code'}, inplace=True)

        chg_proj_merge = pd.merge(chg_proj_2019, chg_proj_2023, how='inner', on='Matrix code')

        occs = set([title[0] for title in sim_occs_list])
        occs = [x.lower() for x in occs]

        sim_occs = chg_proj_merge[chg_proj_merge['2023 National Employment Matrix title'].isin(occs)].drop_duplicates(subset=['2023 National Employment Matrix title'], keep='first')
        return self.create_projection_chart(sim_occs)
    
    def create_heatmap(self, educ_data, occupations, col_name):
        # Drop unnecessary columns to clean the data
        educ_data = educ_data.drop(columns=['2023 National Employment Matrix code'], errors='ignore')

        # Standardize occupation titles and education data for consistency
        educ_data[col_name] = educ_data[col_name].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True).str.strip()
        occupations = [occ.lower().strip() for occ in occupations]

        # Preprocess both strings for comparison
        educ_data[col_name] = educ_data[col_name].apply(lambda x: x.lower().strip())
        occupations = [occ.lower().strip() for occ in occupations]

        # Filter the data to include only relevant occupations
        filtered_data = educ_data[educ_data[col_name].isin(occupations)]
        if filtered_data.empty:
            print("Filtered data is empty. Check the 'occupations' list and 'col_name' values.")
            return go.Figure()

        # Ensure numeric values in education level columns
        education_columns = filtered_data.columns.tolist()[1:]
        filtered_data[education_columns] = filtered_data[education_columns].apply(pd.to_numeric, errors='coerce')
        if filtered_data[education_columns].isna().all().all():
            print("Education level columns have no valid numeric data.")
            return go.Figure()

        # Pivot the data to create a matrix of occupations (rows) and education levels (columns)
        heatmap_data = filtered_data.pivot_table(index=col_name, aggfunc='mean')

        # Extract the occupation names and education level names in the order they appear in the DataFrame
        occupation_labels = heatmap_data.index.tolist()
        education_labels = education_columns

        # Debug print statements to verify labels and data
        print("Occupation Labels:", occupation_labels)
        print("Education Labels:", education_labels)

        # Convert the pivot table to a 2D list for Plotly heatmap
        values = heatmap_data.values.tolist()

        # Create the heatmap using Plotly
        fig = go.Figure(
            data=go.Heatmap(
                z=values,
                x=education_labels,
                y=occupation_labels,
                colorscale='YlGnBu',
                colorbar=dict(title="% Distribution")
            )
        )

        # Update layout for better readability
        fig.update_layout(
            title="Education Distribution Across Occupations",
            xaxis=dict(title="Education Levels"),
            yaxis=dict(title="Occupations"),
            height=600,
            width=1000
        )

        return fig



