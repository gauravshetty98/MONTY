from linkedin_api import Linkedin

class LinkedInProfileFetcher:
    """
    A class to interact with the LinkedIn API to fetch user profiles.

    Attributes:
        email (str): The email used to log in to LinkedIn.
        password (str): The password used to log in to LinkedIn.
    """
    
    def __init__(self, email = 'gze.pois0n@gmail.com', password = 'password'):
        """
        Initializes the LinkedInProfileFetcher with email and password.

        Args:
            email (str): LinkedIn account email.
            password (str): LinkedIn account password.
        """
        self.email = email
        self.password = password
        self.api = Linkedin(self.email, self.password)

    
    def get_profile(self, profile_id):
        """
        Fetches the LinkedIn profile for the given profile ID.

        Args:
            profile_id (str): The profile ID of the LinkedIn user to fetch.

        Returns:
            dict: The profile data retrieved from LinkedIn.
        """
        # if self.api is None:
        #     raise ValueError("You must log in first by calling the 'login' method.")
        
        # Fetch the profile based on profile ID
        profile = self.api.get_profile(profile_id)
        return profile
