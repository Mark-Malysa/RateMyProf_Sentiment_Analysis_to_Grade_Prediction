#!/usr/bin/env python3
"""
Enhanced RateMyProfessor Analysis Pipeline

This is the main entry point for the enhanced analysis pipeline that includes:
- Clean data preprocessing with grade validation
- BERT embeddings for text representation
- Professor/course disentanglement features
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation

Usage:
    python main_enhanced.py                    # Full pipeline with BERT
    python main_enhanced.py --encoder tfidf    # Use TF-IDF instead of BERT
    python main_enhanced.py --fast             # Skip hyperparameter tuning
    python main_enhanced.py --test-mode        # Quick test run

Author: Enhanced for Resume Portfolio
"""

import logging
import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_collection import DataCollector
from src.data.preprocessing import DataPreprocessor
from src.models.sentiment_analysis import SentimentAnalyzer
from src.models.enhanced_grade_predictor import EnhancedGradePredictor
from src.visualization.visualization import DataVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced RateMyProfessor Analysis Pipeline')
    parser.add_argument('--encoder', type=str, default='bert', choices=['bert', 'tfidf'],
                       help='Text encoder type (default: bert)')
    parser.add_argument('--fast', action='store_true',
                       help='Skip hyperparameter tuning for faster execution')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with minimal processing')
    parser.add_argument('--output-dir', type=str, default='visualizations_enhanced',
                       help='Output directory for visualizations')
    return parser.parse_args()


def main():
    """Main pipeline function."""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Enhanced RateMyProfessor Analysis Pipeline")
    logger.info(f"Encoder: {args.encoder.upper()}")
    logger.info(f"Hyperparameter Tuning: {'Disabled' if args.fast else 'Enabled'}")
    logger.info("="*60)
    
    try:
        # ============================================================
        # Step 1: Data Collection with Stratified Split
        # ============================================================
        logger.info("\n[STEP 1/5] Data Collection & Stratified Splitting...")
        
        collector = DataCollector()
        train_df, test_df = collector.load_and_split_data(stratify_by='grade')
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # ============================================================
        # Step 2: Data Preprocessing with Grade Validation
        # ============================================================
        logger.info("\n[STEP 2/5] Data Preprocessing...")
        
        preprocessor = DataPreprocessor()
        
        # Load and preprocess training data
        train_raw = preprocessor.load_data("data/raw/training_reviews.csv")
        processed_train = preprocessor.preprocess_data(
            train_raw, 
            filter_invalid_grades=True,
            add_professor_features=True,
            add_course_features=True
        )
        preprocessor.save_processed_data(processed_train, "data/processed/processed_training_enhanced.csv")
        
        # Load and preprocess test data
        test_raw = preprocessor.load_data("data/raw/test_reviews.csv")
        processed_test = preprocessor.preprocess_data(
            test_raw,
            filter_invalid_grades=True,
            add_professor_features=True,
            add_course_features=True
        )
        preprocessor.save_processed_data(processed_test, "data/processed/processed_test_enhanced.csv")
        
        logger.info(f"Processed training samples: {len(processed_train)}")
        logger.info(f"Processed test samples: {len(processed_test)}")
        
        # ============================================================
        # Step 3: Sentiment Analysis with BERT/TF-IDF
        # ============================================================
        logger.info(f"\n[STEP 3/5] Sentiment Analysis ({args.encoder.upper()})...")
        
        sentiment_analyzer = SentimentAnalyzer(encoder_type=args.encoder)
        
        # Prepare data
        X_train, y_train = sentiment_analyzer.prepare_data(processed_train)
        
        # Train model with cross-validation
        sentiment_metrics = sentiment_analyzer.train_model(
            X_train, y_train, 
            model_type='lr',  # Use LR for BERT (NB doesn't work with negative values)
            use_cv=True,
            cv_folds=5
        )
        
        # Evaluate on test set
        X_test, y_test = sentiment_analyzer.prepare_data(processed_test)
        test_sentiment_metrics = sentiment_analyzer.evaluate_model(X_test, y_test)
        
        # Add sentiment scores to dataframes
        logger.info("Adding sentiment scores to dataframes...")
        processed_train['sentiment_score'] = sentiment_analyzer.get_sentiment_scores(
            processed_train['cleaned_text'].fillna('').tolist()
        )
        processed_test['sentiment_score'] = sentiment_analyzer.get_sentiment_scores(
            processed_test['cleaned_text'].fillna('').tolist()
        )
        
        # Save sentiment model
        sentiment_analyzer.save_model(
            "models/sentiment_model_enhanced.pkl",
            "models/encoder_enhanced.pkl"
        )
        
        # ============================================================
        # Step 4: Grade Prediction with Enhanced Predictor
        # ============================================================
        logger.info(f"\n[STEP 4/5] Grade Prediction (BERT + Hyperparameter Tuning)...")
        
        predictor = EnhancedGradePredictor(
            use_bert=(args.encoder == 'bert'),
            use_hyperparameter_tuning=not args.fast
        )
        
        # Train with hyperparameter tuning
        grade_metrics = predictor.train(
            processed_train,
            cv_folds=5,
            use_fast_tuning=True  # Use fast tuning even with full mode
        )
        
        # Evaluate on test set
        test_grade_metrics = predictor.evaluate(processed_test)
        
        # Save model
        predictor.save_model("models/grade_predictor_enhanced.pkl")
        
        # Get and save feature importance
        importance_df = predictor.get_feature_importance()
        importance_df.to_csv("models/feature_importance_enhanced.csv", index=False)
        
        # Example prediction
        sample_features = {
            'rating': 4.5,
            'difficulty': 3.0,
            'cleaned_text': "The professor was very helpful and explained concepts clearly.",
            'professor_avg_rating': 4.2,
            'professor_avg_difficulty': 2.8
        }
        prediction = predictor.predict_grade(sample_features)
        logger.info(f"\nExample Prediction:")
        logger.info(f"  Predicted Grade: {prediction['predicted_grade']}")
        logger.info(f"  Predicted GPA: {prediction['predicted_gpa']}")
        if 'confidence_interval' in prediction:
            ci = prediction['confidence_interval']
            logger.info(f"  Confidence Interval: {ci['lower_grade']}-{ci['upper_grade']} "
                       f"({ci['lower_gpa']}-{ci['upper_gpa']} GPA)")
        
        # ============================================================
        # Step 5: Visualization
        # ============================================================
        logger.info(f"\n[STEP 5/5] Creating Visualizations...")
        
        os.makedirs(args.output_dir, exist_ok=True)
        visualizer = DataVisualizer(output_dir=args.output_dir)
        
        # Create visualizations
        visualizer.plot_sentiment_distribution(processed_train, suffix="_train")
        visualizer.plot_grade_correlation(processed_train, suffix="_train")
        visualizer.plot_difficulty_rating(processed_train, suffix="_train")
        visualizer.plot_feature_importance(importance_df)
        
        # Create correlation matrix with enhanced features
        feature_cols = ['rating', 'difficulty', 'sentiment_score', 'grade_numerical']
        if 'professor_avg_rating' in processed_train.columns:
            feature_cols.append('professor_avg_rating')
        if 'course_avg_difficulty' in processed_train.columns:
            feature_cols.append('course_avg_difficulty')
        
        available_cols = [c for c in feature_cols if c in processed_train.columns]
        if len(available_cols) >= 2:
            visualizer.create_correlation_matrix(processed_train, available_cols, suffix="_enhanced")
        
        # ============================================================
        # Final Report
        # ============================================================
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE - RESULTS SUMMARY")
        logger.info("="*60)
        
        logger.info("\n📊 Sentiment Analysis:")
        logger.info(f"   Encoder: {args.encoder.upper()}")
        logger.info(f"   Test Accuracy: {test_sentiment_metrics['accuracy']:.4f}")
        logger.info(f"   Test F1 Score: {test_sentiment_metrics['f1_score']:.4f}")
        
        logger.info("\n📈 Grade Prediction:")
        logger.info(f"   Test RMSE: {test_grade_metrics['rmse']:.4f}")
        logger.info(f"   Test MAE: {test_grade_metrics['mae']:.4f}")
        logger.info(f"   Test R²: {test_grade_metrics['r2']:.4f}")
        
        if predictor.best_params:
            logger.info(f"\n🔧 Best Hyperparameters:")
            for param, value in predictor.best_params.items():
                logger.info(f"   {param}: {value}")
        
        logger.info(f"\n📁 Outputs:")
        logger.info(f"   Models: models/sentiment_model_enhanced.pkl, models/grade_predictor_enhanced.pkl")
        logger.info(f"   Visualizations: {args.output_dir}/")
        logger.info(f"   Feature Importance: models/feature_importance_enhanced.csv")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Enhanced pipeline completed successfully!")
        logger.info("="*60)
        
        return {
            'sentiment_metrics': sentiment_metrics,
            'test_sentiment_metrics': test_sentiment_metrics,
            'grade_metrics': grade_metrics,
            'test_grade_metrics': test_grade_metrics
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
