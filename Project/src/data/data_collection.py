import pandas as pd
import json
import logging
import os
from typing import List, Dict, Any
import ast

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.base_url = "https://www.ratemyprofessors.com"
    
    def load_existing_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load existing review data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of review dictionaries
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Flatten the nested structure
            reviews = []
            for course_reviews in data:
                reviews.extend(course_reviews)
            
            logger.info(f"Successfully loaded {len(reviews)} reviews from {filepath}")
            return reviews
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def parse_json_string(self, json_str):
        """
        Parse a JSON string into a dictionary.
        
        Args:
            json_str: JSON string to parse
            
        Returns:
            Dictionary containing the parsed data
        """
        try:
            if pd.isna(json_str):
                return {}
            # Use ast.literal_eval to safely evaluate the string representation of a dictionary
            return ast.literal_eval(json_str)
        except:
            return {}
    
    def save_data(self, data: List[Dict[str, Any]], filepath: str):
        """
        Save data to a CSV file.
        
        Args:
            data: List of review dictionaries
            filepath: Output file path
        """
        try:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)
            
            # Clean and standardize columns
            df = df.rename(columns={
                'Quality': 'rating',
                'Difficulty': 'difficulty',
                'Grade': 'grade',
                'Comment': 'text',
                'professor': 'professor_name'
            })
            
            # Convert numeric columns
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce')
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(df)} reviews to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def load_and_split_data(self):
        """
        Load existing data and split into training and test sets.
        """
        try:
            # Load data
            data = self.load_existing_data("datasets/all_reviews.json")
            
            # Split into training and test sets (80-20 split)
            train_size = int(len(data) * 0.8)
            training_data = data[:train_size]
            test_data = data[train_size:]
            
            # Save the splits
            self.save_data(training_data, "data/raw/training_reviews.csv")
            self.save_data(test_data, "data/raw/test_reviews.csv")
            
        except Exception as e:
            logger.error(f"Error in load_and_split_data: {e}")
            raise

def main():
    # Example usage
    collector = DataCollector()
    collector.load_and_split_data()

if __name__ == "__main__":
    main() 