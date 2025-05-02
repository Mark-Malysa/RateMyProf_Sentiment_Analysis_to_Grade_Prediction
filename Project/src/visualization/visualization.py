import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, suffix: str = ""):
        """
        Plot distribution of sentiment scores.
        
        Args:
            df: DataFrame containing sentiment scores
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='sentiment_score', bins=20)
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.savefig(f"{self.output_dir}/sentiment_distribution{suffix}.png")
        plt.close()
    
    def plot_grade_correlation(self, df: pd.DataFrame, suffix: str = ""):
        """
        Plot correlation between sentiment and grades.
        
        Args:
            df: DataFrame containing sentiment scores and grades
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='sentiment_score', y='grade_numerical')
        plt.title('Correlation between Sentiment and Grades')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Grade')
        plt.savefig(f"{self.output_dir}/grade_correlation{suffix}.png")
        plt.close()
    
    def plot_difficulty_rating(self, df: pd.DataFrame, suffix: str = ""):
        """
        Plot relationship between difficulty and rating.
        
        Args:
            df: DataFrame containing difficulty and rating scores
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='difficulty', y='rating')
        plt.title('Relationship between Difficulty and Rating')
        plt.xlabel('Difficulty')
        plt.ylabel('Rating')
        plt.savefig(f"{self.output_dir}/difficulty_rating{suffix}.png")
        plt.close()
    
    def plot_department_comparison(self, df: pd.DataFrame, suffix: str = ""):
        """
        Plot comparison of ratings across departments.
        
        Args:
            df: DataFrame containing department and rating information
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='department', y='rating')
        plt.title('Rating Distribution by Department')
        plt.xlabel('Department')
        plt.ylabel('Rating')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/department_comparison{suffix}.png")
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, suffix: str = ""):
        """
        Plot feature importance scores.
        
        Args:
            importance_df: DataFrame containing feature importance scores
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance for Grade Prediction')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance{suffix}.png")
        plt.close()
    
    def plot_sentiment_trends(self, df: pd.DataFrame, suffix: str = ""):
        """
        Plot sentiment trends over time.
        
        Args:
            df: DataFrame containing sentiment scores and dates
            suffix: Suffix to add to output filename
        """
        # Convert date column to datetime if needed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=df, x='date', y='sentiment_score')
            plt.title('Sentiment Trends Over Time')
            plt.xlabel('Date')
            plt.ylabel('Average Sentiment Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/sentiment_trends{suffix}.png")
            plt.close()
    
    def create_correlation_matrix(self, df: pd.DataFrame, features: List[str], suffix: str = ""):
        """
        Create correlation matrix heatmap.
        
        Args:
            df: DataFrame containing features
            features: List of features to include in correlation matrix
            suffix: Suffix to add to output filename
        """
        plt.figure(figsize=(10, 8))
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix{suffix}.png")
        plt.close()
    
    def plot_model_comparison(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float], 
                            model_name: str, suffix: str = ""):
        """
        Plot comparison of model performance on training and test data.
        
        Args:
            train_metrics: Dictionary of training metrics
            test_metrics: Dictionary of test metrics
            model_name: Name of the model
            suffix: Suffix to add to output filename
        """
        metrics = ['accuracy', 'f1_score', 'rmse', 'r2_score']
        train_values = [train_metrics.get(m, 0) for m in metrics]
        test_values = [test_metrics.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, train_values, width, label='Training')
        plt.bar(x + width/2, test_values, width, label='Test')
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(f'{model_name} Performance Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison_{model_name}{suffix}.png")
        plt.close()

def main():
    # Example usage
    visualizer = DataVisualizer()
    
    # Load processed data
    training_df = pd.read_csv("data/processed/processed_training_reviews.csv")
    test_df = pd.read_csv("data/processed/processed_test_reviews.csv")
    
    # Create visualizations for training data
    visualizer.plot_sentiment_distribution(training_df, suffix="_train")
    visualizer.plot_grade_correlation(training_df, suffix="_train")
    visualizer.plot_difficulty_rating(training_df, suffix="_train")
    visualizer.plot_department_comparison(training_df, suffix="_train")
    
    # Create visualizations for test data
    visualizer.plot_sentiment_distribution(test_df, suffix="_test")
    visualizer.plot_grade_correlation(test_df, suffix="_test")
    visualizer.plot_difficulty_rating(test_df, suffix="_test")
    
    # Load feature importance data
    importance_df = pd.read_csv("models/feature_importance.csv")
    visualizer.plot_feature_importance(importance_df)
    
    # Create correlation matrices
    features = ['rating', 'difficulty', 'sentiment_score', 'grade_numerical']
    visualizer.create_correlation_matrix(training_df, features, suffix="_train")
    visualizer.create_correlation_matrix(test_df, features, suffix="_test")
    
    # Plot sentiment trends if date data is available
    if 'date' in training_df.columns:
        visualizer.plot_sentiment_trends(training_df, suffix="_train")
    if 'date' in test_df.columns:
        visualizer.plot_sentiment_trends(test_df, suffix="_test")

if __name__ == "__main__":
    main() 