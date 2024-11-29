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
        occs = set([title[0] for title in sim_occs_list])
        salary_data = data[data[title_col].isin(occs)].drop_duplicates(subset=[title_col], keep='first')

        hourly_sals = list(salary_data[col_1])
        occ_titles = list(salary_data[title_col])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=occ_titles,
            y=hourly_sals,
            name=label_1,
            marker_color='skyblue'
        ))

        fig.update_layout(
            title=plt_title,
            xaxis=dict(title='Occupation Titles', tickangle=45, automargin=True),
            yaxis=dict(title=label_1, titlefont=dict(color='skyblue'), tickfont=dict(color='skyblue')),
            height=600,
            width=1000
        )
        return fig

    def double_bar_plot(self, sim_occs_list, data_1, data_2, col_1, col_2, label_1, label_2, title_col_1, title_col_2, plt_title):
        fig = self.single_bar_plot(sim_occs_list, data_1, col_1, label_1, title_col_1, plt_title)

        occs = set([title[0] for title in sim_occs_list])
        subset_data = data_2[data_2[title_col_2].isin(occs)].drop_duplicates(subset=[title_col_2], keep='first')

        col_data_2 = list(subset_data[col_2])
        subset_titles = list(subset_data[title_col_2])

        fig.add_trace(go.Bar(
            x=subset_titles,
            y=col_data_2,
            name=label_2,
            marker_color='salmon'
        ))

        fig.update_layout(barmode='group')
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
        occs = set([title[0] for title in sim_occs_list])
        occs = [x.lower() for x in occs]

        educ_data[col_name] = educ_data[col_name].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
        subset_educ_data = educ_data[educ_data[col_name].isin(occs)].drop_duplicates(subset = [col_name], keep = 'first')
        
        ep = EmbeddingProcessor(tokenizer, model)
        embeddings = ep.load_embeddings(list(educ_data.columns), 32, filename="edu_embeddings.npy")
        eq_degree = ep.get_similar_items(user_education, embeddings)

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
