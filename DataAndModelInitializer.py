import pandas as pd
from transformers import AutoTokenizer, AutoModel

class DataAndModelInitializer:
    """
    A class that handles initialization of dataframes and model loading.
    
    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModel): The pre-trained model.
    """
    
    def __init__(self):
        """
        Initializes the DataAndModelInitializer object.
        """
        self.tokenizer = None
        self.model = None
        self.dataframes = None
    
    def initialize_dataframes(self):
        """
        Initializes dataframes by reading Excel files for different datasets.
        
        Returns:
            dict: A dictionary containing the initialized dataframes.
        """
        # Reading the 2023 data
        data_2023 = pd.read_excel('project_data/oesm23nat/national_M2023_dl.xlsx')
        
        # Reading the 2019 data
        data_2019 = pd.read_excel('project_data/oesm19nat/national_M2019_dl.xlsx')
        
        # Reading the 2033 industry data (from a specific sheet and skipping rows)
        ind_data_2033 = pd.read_excel('project_data/2023-33/industry.xlsx', sheet_name=11, skiprows=1)

        educ_data_2023 = pd.read_excel('project_data/2023-33/education.xlsx', sheet_name=3, skiprows=1)

        chg_proj_2019 = pd.read_excel('project_data/2019-29/occupation.xlsx', sheet_name=8, skiprows=1)
        chg_proj_2023 = pd.read_excel('project_data/2023-33/occupation.xlsx', sheet_name=10, skiprows=1)

        occ_ed_req_19 = pd.read_excel('project_data/extra/education_2019.xlsx', sheet_name = 6, skiprows = 4)
        occ_ed_req_23 = pd.read_excel('project_data/extra/education_2023.xlsx', sheet_name = 6, skiprows = 4)

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

        # Replace the column values
        occ_ed_req_23['Education Level Numeric'] = occ_ed_req_23['Typical entry-level educational requirement'].map(education_mapping)
        occ_ed_req_19['Education Level Numeric'] = occ_ed_req_19['Typical entry-level educational requirement'].map(education_mapping)
        
        # Store dataframes as a dictionary in the instance variable
        self.dataframes = {
            'data_2023': data_2023,
            'data_2019': data_2019,
            'ind_data_2033': ind_data_2033,
            'educ_data_2023': educ_data_2023,
            'chg_proj_2019': chg_proj_2019,
            'chg_proj_2023': chg_proj_2023,
            'occ_ed_req_19': occ_ed_req_19,
            'occ_ed_req_23': occ_ed_req_23
        }
        return self.dataframes
    
    def init_model(self):
        """
        Initializes and loads the tokenizer and model.
        
        Returns:
            tuple: A tuple containing the initialized tokenizer and model.
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        return self.tokenizer, self.model
