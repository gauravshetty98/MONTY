# MONTY
MONTY is a tool which leverages BLS data and uses big data algorithms to present different types of analysis based on the users industry

The main idea behind the project is to create a website where the user can just enter their linkedin profile based on which we show different graphs based on the users education, experience, industry, etc. The analysis should help the user answer questions like - Is my industry growing or declining? What are the main locations where my industry is booming? What is the general expectation for education and experience in my industry? Based on education and experience how is the salary increasing? and other questions similar to this. 

To get similar jobs we are making use of cosine similarity and BERT. The code you can find in the "EmbeddingProcessor.py" file. Then based on the result we try to get 5-7 jobs/industry/occupation/education based on the users profile and then show the graphs.
Here is a short brief of the different files present in the github repo:

DataAndModelInitializer.py - To initialize the BERT model. To get the necessary dataframes from the professors dataset.
LinkedInProfileFetcher.py - To fetch the users LinkedIn profile details.
VisualizationTools.py - To create different kinds of plots.
EmbeddingProcessor.py - To get occupations/education present in the BLS data that are similar to the user's qualification
app.py  - The main file which calls the different functions present in these classes and the uses the plotly dash method to display the graphs in a webpage.
