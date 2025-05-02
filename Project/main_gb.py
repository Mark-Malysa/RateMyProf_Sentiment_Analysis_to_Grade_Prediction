import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.data_collection import DataCollector
from src.data.preprocessing import DataPreprocessor
from src.models.sentiment_analysis import SentimentAnalyzer
from src.models.predictive_modeling_gb import GradePredictorGB
from src.visualization.visualization import DataVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_sentiment_scores(df: pd.DataFrame, sentiment_analyzer: SentimentAnalyzer) -> pd.DataFrame:
    X = sentiment_analyzer.vectorizer.transform(df['cleaned_text'])
    probabilities = sentiment_analyzer.model.predict_proba(X)
    df['sentiment_score'] = probabilities[:, 1]
    return df

def main():
    try:
        # Step 1: Data Collection
        logger.info("Starting data collection...")
        logger.info("Loading and splitting data...")
        collector = DataCollector()
        collector.load_and_split_data()
        
        # Step 2: Data Preprocessing
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        
        # Preprocess training data
        logger.info("Preprocessing training data...")
        train_df = preprocessor.load_data("data/raw/training_reviews.csv")
        processed_train_df = preprocessor.preprocess_data(train_df)
        preprocessor.save_processed_data(processed_train_df, "data/processed/processed_training_reviews.csv")
        
        # Preprocess test data
        logger.info("Preprocessing test data...")
        test_df = preprocessor.load_data("data/raw/test_reviews.csv")
        processed_test_df = preprocessor.preprocess_data(test_df)
        preprocessor.save_processed_data(processed_test_df, "data/processed/processed_test_reviews.csv")
        
        # Step 3: Sentiment Analysis
        logger.info("Starting sentiment analysis...")
        sentiment_analyzer = SentimentAnalyzer()
        
        # Train sentiment model
        logger.info("Training sentiment model...")
        X_train, y_train = sentiment_analyzer.prepare_data(processed_train_df)
        metrics = sentiment_analyzer.train_model(X_train, y_train, model_type='nb')
        sentiment_analyzer.save_model("models/sentiment_model.pkl", "models/vectorizer.pkl")
        
        # Evaluate sentiment model
        logger.info("Evaluating sentiment model on test data...")
        X_test, y_test = sentiment_analyzer.prepare_data(processed_test_df)
        test_metrics = sentiment_analyzer.evaluate_model(X_test, y_test)
        
        # Add sentiment scores to DataFrames
        logger.info("Adding sentiment scores to DataFrames...")
        processed_train_df = add_sentiment_scores(processed_train_df, sentiment_analyzer)
        processed_test_df = add_sentiment_scores(processed_test_df, sentiment_analyzer)
        
        # Step 4: Grade Prediction (Gradient Boosting)
        logger.info("Starting grade prediction modeling (Gradient Boosting)...")
        predictor = GradePredictorGB()
        
        # Train grade prediction model
        logger.info("Training grade prediction model...")
        predictor.train(processed_train_df)
        
        # Evaluate grade prediction model
        logger.info("Evaluating grade prediction model...")
        metrics = predictor.evaluate_model(processed_test_df)
        
        # Save the model
        predictor.save_model("models/grade_predictor_gb.pkl")
        
        # Example prediction
        sample_features = {
            'rating': 4.5,
            'difficulty': 3.0,
            'cleaned_text': "The professor was very helpful and explained concepts clearly."
        }
        
        predicted_grade, confidence_interval = predictor.predict_grade(sample_features)
        logger.info(f"Example Prediction:")
        logger.info(f"Predicted grade: {predicted_grade}")
        logger.info(f"95% Confidence Interval: {confidence_interval}")
        
        # Save feature importance
        importance_df = predictor.get_feature_importance()
        importance_df.to_csv("models/feature_importance_gb.csv", index=False)
        
        # Step 5: Visualization
        logger.info("Creating visualizations...")
        visualizer = DataVisualizer(output_dir="visualizations_gb")
        
        # Visualize training data
        logger.info("Visualizing training data...")
        visualizer.plot_sentiment_distribution(processed_train_df)
        visualizer.plot_grade_correlation(processed_train_df)
        visualizer.plot_difficulty_rating(processed_train_df)
        visualizer.plot_department_comparison(processed_train_df)
        visualizer.plot_feature_importance(importance_df)
        
        # Visualize test data
        logger.info("Visualizing test data...")
        visualizer.plot_sentiment_distribution(processed_test_df, suffix="_test")
        visualizer.plot_grade_correlation(processed_test_df, suffix="_test")
        visualizer.plot_difficulty_rating(processed_test_df, suffix="_test")
        
        # Create correlation matrices
        features = ['rating', 'difficulty', 'sentiment_score', 'grade_numerical']
        visualizer.create_correlation_matrix(processed_train_df, features, suffix="_train")
        visualizer.create_correlation_matrix(processed_test_df, features, suffix="_test")
        
        if 'date' in processed_train_df.columns:
            visualizer.plot_sentiment_trends(processed_train_df, suffix="_train")
        if 'date' in processed_test_df.columns:
            visualizer.plot_sentiment_trends(processed_test_df, suffix="_test")
        
        logger.info("Analysis pipeline (Gradient Boosting) completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 