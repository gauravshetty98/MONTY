import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine
import os

class EmbeddingProcessor:
    """
    A class to handle mean pooling, batch embedding computation, similarity calculation,
    and saving/loading embeddings.
    
    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the model.
        model (AutoModel): The pre-trained model for generating embeddings.
    """

    def __init__(self, tokenizer, model):
        """
        Initializes the EmbeddingProcessor with a tokenizer and model.
        
        Args:
            tokenizer (AutoTokenizer): The tokenizer used to tokenize input texts.
            model (AutoModel): The pre-trained model used to generate embeddings.
        """
        self.tokenizer = tokenizer
        self.model = model
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings, weighted by the attention mask.

        Args:
            token_embeddings (Tensor): The token embeddings from the model.
            attention_mask (Tensor): The attention mask for the input.

        Returns:
            Tensor: The pooled embeddings.
        """
        token_embeddings = token_embeddings * attention_mask.unsqueeze(-1)
        return token_embeddings.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)

    def batch_compute_embeddings(self, col_vals, batch_size=32):
        """
        Compute embeddings for a list of values in batches.

        Args:
            col_vals (list): List of text values to compute embeddings for.
            batch_size (int): The size of each batch for processing.

        Returns:
            dict: A dictionary mapping text to its computed embedding.
        """
        embeddings = {}
        dataloader = DataLoader(col_vals, batch_size=batch_size, shuffle=False)
        
        for batch in dataloader:
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=64)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            for i, title in enumerate(batch):
                embeddings[title] = batch_embeddings[i].cpu().numpy()
        
        return embeddings

    def get_similar_items(self, ind_title, cached_embeddings):
        """
        Compute similarity between a given industry title and cached embeddings.

        Args:
            ind_title (str): The industry title to compare against cached embeddings.
            cached_embeddings (dict): A dictionary of precomputed embeddings.

        Returns:
            list: The top 10 most similar items from the cached embeddings.
        """
        ind_inputs = self.tokenizer(ind_title, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            ind_outputs = self.model(**ind_inputs)
        
        ind_embedding = self.mean_pooling(ind_outputs.last_hidden_state, ind_inputs['attention_mask'])
        ind_vector = ind_embedding.squeeze().cpu().numpy()
        
        similarities = [
            (title, 1 - cosine(ind_vector, occ_vector)) 
            for title, occ_vector in cached_embeddings.items()
        ]
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:10]

    def save_embeddings(self, embeddings, filename="embeddings.npy"):
        """
        Save computed embeddings to a file.

        Args:
            embeddings (dict): The dictionary of embeddings to save.
            filename (str): The filename where embeddings will be saved.
        """
        np.save(filename, embeddings)

    def load_embeddings(self, string_list, batch_size=64, filename="embeddings.npy"):
        """
        Load embeddings from a file, or compute and save them if they don't exist.

        Args:
            string_list (list): The list of text to compute embeddings for.
            batch_size (int): The size of each batch for processing.
            filename (str): The filename to load or save embeddings.

        Returns:
            dict: The embeddings loaded or newly computed.
        """
        if os.path.exists(filename):
            embeddings = np.load(filename, allow_pickle=True).item()
        else:
            embeddings = self.batch_compute_embeddings(string_list, batch_size)
            self.save_embeddings(embeddings, filename)
        
        return embeddings
