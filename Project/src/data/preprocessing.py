import pandas as pd
import numpy as np
from typing import Tuple
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('wordnet')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the raw data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the raw data
        """
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Successfully loaded data from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into words
        words = text.split()
        
        # Remove stopwords and lemmatize
        cleaned_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]
        
        return ' '.join(cleaned_words)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the entire dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Clean text data
        if 'text' in df_processed.columns:
            df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        # Handle missing values
        df_processed = df_processed.dropna(subset=['rating', 'difficulty'])
        
        # Convert grade to numerical values if needed
        if 'grade' in df_processed.columns:
            grade_mapping = {
                'A+': 4.0, 'A': 4.0, 'A-': 3.7,
                'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7,
                'D+': 1.3, 'D': 1.0, 'D-': 0.7,
                'F': 0.0
            }
            df_processed['grade_numerical'] = df_processed['grade'].map(grade_mapping)
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """
        Save processed data to CSV file.
        
        Args:
            df: Processed DataFrame
            filepath: Output file path
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully saved processed data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise

def main():
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load raw data
    raw_data = preprocessor.load_data("data/raw/professor_reviews.csv")
    
    # Preprocess data
    processed_data = preprocessor.preprocess_data(raw_data)
    
    # Save processed data
    preprocessor.save_processed_data(processed_data, "data/processed/processed_reviews.csv")

if __name__ == "__main__":
    main() 