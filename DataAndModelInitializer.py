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
        
        # Store dataframes as a dictionary in the instance variable
        self.dataframes = {
            'data_2023': data_2023,
            'data_2019': data_2019,
            'ind_data_2033': ind_data_2033,
            'educ_data_2023': educ_data_2023,
            'chg_proj_2019': chg_proj_2019,
            'chg_proj_2023': chg_proj_2023
        }
        return self.dataframes
    
    def init_model(self):
        """
        Initializes and loads the tokenizer and model.
        
        Returns:
            tuple: A tuple containing the initialized tokenizer and model.
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("jjzha/jobbert-base-cased")
        self.model = AutoModel.from_pretrained("jjzha/jobbert-base-cased")
        return self.tokenizer, self.model
