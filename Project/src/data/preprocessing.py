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
    """Enhanced data preprocessor with grade validation and feature extraction."""
    
    # Valid letter grades that can be mapped to numerical values
    VALID_GRADES = {'A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F'}
    
    # Invalid grade values to filter out
    INVALID_GRADES = {
        'Not sure yet', 'Not_Sure_Yet',
        'Rather not say', 'Rather_Not_Say',
        'Incomplete', 'Pass', 'Drop/Withdrawal',
        'Audit/No Grade', 'WIthdrew', 'Withdrew', 'INC'
    }
    
    # Complete grade mapping including all edge cases
    GRADE_MAPPING = {
        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
        'F': 0.0
    }
    
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Expose grade_mapping as instance attribute for external access
        self.grade_mapping = self.GRADE_MAPPING.copy()
        
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
    
    def validate_grade(self, grade) -> bool:
        """
        Check if a grade value is valid for training.
        
        Args:
            grade: Grade value to validate
            
        Returns:
            True if grade is valid, False otherwise
        """
        if pd.isna(grade):
            return False
        grade_str = str(grade).strip()
        return grade_str in self.VALID_GRADES
    
    def clean_grade(self, grade) -> str:
        """
        Normalize and clean grade value.
        
        Args:
            grade: Raw grade value
            
        Returns:
            Cleaned grade string or None if invalid
        """
        if pd.isna(grade):
            return None
        grade_str = str(grade).strip()
        
        # Check if it's a valid grade
        if grade_str in self.VALID_GRADES:
            return grade_str
        
        # Check if it's explicitly invalid
        if grade_str in self.INVALID_GRADES:
            return None
            
        return None
    
    def add_professor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add professor-level aggregated features for disentanglement.
        
        Features added:
        - professor_avg_rating: Average rating for this professor
        - professor_avg_difficulty: Average difficulty for this professor
        - professor_review_count: Number of reviews for this professor
        - professor_avg_grade: Average grade given by students for this professor
        
        Args:
            df: DataFrame with professor_name column
            
        Returns:
            DataFrame with professor features added
        """
        if 'professor_name' not in df.columns:
            logger.warning("No professor_name column found, skipping professor features")
            return df
            
        df_result = df.copy()
        
        # Calculate professor-level statistics
        prof_stats = df_result.groupby('professor_name').agg({
            'rating': ['mean', 'count', 'std'],
            'difficulty': ['mean', 'std']
        }).reset_index()
        
        prof_stats.columns = [
            'professor_name', 
            'professor_avg_rating', 'professor_review_count', 'professor_rating_std',
            'professor_avg_difficulty', 'professor_difficulty_std'
        ]
        
        # Add professor grade average if grade_numerical exists
        if 'grade_numerical' in df_result.columns:
            grade_stats = df_result.groupby('professor_name')['grade_numerical'].agg(['mean', 'std']).reset_index()
            grade_stats.columns = ['professor_name', 'professor_avg_grade', 'professor_grade_std']
            prof_stats = prof_stats.merge(grade_stats, on='professor_name', how='left')
        
        # Merge back to main dataframe
        df_result = df_result.merge(prof_stats, on='professor_name', how='left')
        
        # Fill NaN std values with 0 (single review professors)
        std_cols = [c for c in df_result.columns if c.endswith('_std')]
        df_result[std_cols] = df_result[std_cols].fillna(0)
        
        logger.info(f"Added professor features for {len(prof_stats)} unique professors")
        return df_result
    
    def add_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add course-level baseline features for disentanglement.
        
        Features added:
        - course_avg_rating: Average rating for this course
        - course_avg_difficulty: Average difficulty for this course
        - course_avg_grade: Baseline grade for this course
        
        Args:
            df: DataFrame with course_id column
            
        Returns:
            DataFrame with course features added
        """
        if 'course_id' not in df.columns:
            logger.warning("No course_id column found, skipping course features")
            return df
            
        df_result = df.copy()
        
        # Clean course_id (remove leading spaces)
        df_result['course_id'] = df_result['course_id'].str.strip()
        
        # Calculate course-level statistics
        course_stats = df_result.groupby('course_id').agg({
            'rating': 'mean',
            'difficulty': 'mean'
        }).reset_index()
        
        course_stats.columns = ['course_id', 'course_avg_rating', 'course_avg_difficulty']
        
        # Add course grade average if grade_numerical exists
        if 'grade_numerical' in df_result.columns:
            grade_stats = df_result.groupby('course_id')['grade_numerical'].mean().reset_index()
            grade_stats.columns = ['course_id', 'course_avg_grade']
            course_stats = course_stats.merge(grade_stats, on='course_id', how='left')
        
        # Merge back to main dataframe
        df_result = df_result.merge(course_stats, on='course_id', how='left')
        
        logger.info(f"Added course features for {len(course_stats)} unique courses")
        return df_result
    
    def preprocess_data(self, df: pd.DataFrame, filter_invalid_grades: bool = True,
                       add_professor_features: bool = True, 
                       add_course_features: bool = True) -> pd.DataFrame:
        """
        Preprocess the entire dataset with enhanced cleaning and features.
        
        Args:
            df: Raw DataFrame
            filter_invalid_grades: If True, remove rows with invalid grade values
            add_professor_features: If True, add professor-level features
            add_course_features: If True, add course-level features
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        initial_count = len(df_processed)
        
        # Clean text data
        if 'text' in df_processed.columns:
            df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        # Handle missing values in required columns
        df_processed = df_processed.dropna(subset=['rating', 'difficulty'])
        
        # Clean and validate grades
        if 'grade' in df_processed.columns:
            # Clean grade values
            df_processed['grade_cleaned'] = df_processed['grade'].apply(self.clean_grade)
            
            # Map to numerical values
            df_processed['grade_numerical'] = df_processed['grade_cleaned'].map(self.GRADE_MAPPING)
            
            # Log grade statistics
            valid_grades = df_processed['grade_numerical'].notna().sum()
            logger.info(f"Valid grades: {valid_grades}/{len(df_processed)} ({100*valid_grades/len(df_processed):.1f}%)")
            
            if filter_invalid_grades:
                # Only keep rows with valid grades for grade prediction training
                df_processed = df_processed[df_processed['grade_numerical'].notna()]
                logger.info(f"Filtered to {len(df_processed)} rows with valid grades (from {initial_count})")
        
        # Add professor features (for disentanglement)
        if add_professor_features:
            df_processed = self.add_professor_features(df_processed)
        
        # Add course features (for baseline)
        if add_course_features:
            df_processed = self.add_course_features(df_processed)
        
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