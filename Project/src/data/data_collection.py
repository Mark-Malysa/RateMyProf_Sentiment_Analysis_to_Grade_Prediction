import pandas as pd
import json
import logging
import os
from typing import List, Dict, Any, Optional
import ast
from sklearn.model_selection import train_test_split

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
    
    def load_and_split_data(self, stratify_by: str = 'grade', test_size: float = 0.2, 
                             random_state: int = 42):
        """
        Load existing data and split into training and test sets with stratification.
        
        Args:
            stratify_by: Column to stratify by (default: 'grade')
            test_size: Proportion of data for test set (default: 0.2)
            random_state: Random seed for reproducibility
        """
        try:
            # Load data
            data = self.load_existing_data("datasets/all_reviews.json")
            
            # Convert to DataFrame for stratified splitting
            df = pd.DataFrame(data)
            
            # Clean column names for stratification
            if 'Grade' in df.columns:
                df['grade_for_strat'] = df['Grade'].fillna('NO_GRADE')
            else:
                df['grade_for_strat'] = 'NO_GRADE'
            
            # Filter out rare grades for stratification (need at least 2 samples per class)
            grade_counts = df['grade_for_strat'].value_counts()
            valid_strat_grades = grade_counts[grade_counts >= 2].index
            df['grade_for_strat'] = df['grade_for_strat'].apply(
                lambda x: x if x in valid_strat_grades else 'OTHER'
            )
            
            try:
                # Stratified split
                train_df, test_df = train_test_split(
                    df, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=df['grade_for_strat']
                )
                logger.info(f"Performed stratified split by {stratify_by}")
            except ValueError as e:
                # Fallback to random split if stratification fails
                logger.warning(f"Stratification failed ({e}), falling back to random split")
                train_df, test_df = train_test_split(
                    df, 
                    test_size=test_size, 
                    random_state=random_state
                )
            
            # Drop the temporary stratification column
            train_df = train_df.drop(columns=['grade_for_strat'])
            test_df = test_df.drop(columns=['grade_for_strat'])
            
            # Convert back to list of dicts for save_data
            training_data = train_df.to_dict('records')
            test_data = test_df.to_dict('records')
            
            # Log split statistics
            logger.info(f"Training set: {len(training_data)} reviews")
            logger.info(f"Test set: {len(test_data)} reviews")
            
            # Log grade distribution comparison
            if 'Grade' in train_df.columns:
                train_grades = train_df['Grade'].value_counts(normalize=True).head(5)
                test_grades = test_df['Grade'].value_counts(normalize=True).head(5)
                logger.info(f"Training grade distribution (top 5): {train_grades.to_dict()}")
                logger.info(f"Test grade distribution (top 5): {test_grades.to_dict()}")
            
            # Save the splits
            self.save_data(training_data, "data/raw/training_reviews.csv")
            self.save_data(test_data, "data/raw/test_reviews.csv")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error in load_and_split_data: {e}")
            raise

def main():
    # Example usage
    collector = DataCollector()
    collector.load_and_split_data()

if __name__ == "__main__":
    main() 