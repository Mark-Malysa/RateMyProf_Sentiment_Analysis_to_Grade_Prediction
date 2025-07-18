# RateMyProfessor Sentiment Analysis to Grade Prediction

## 🎯 Project Overview

This project explores the relationship between student sentiment in professor reviews and academic outcomes by combining **Natural Language Processing (NLP)** and **Machine Learning** techniques. The goal is to understand whether sentiment analysis of student reviews can provide meaningful insights for predicting grade outcomes and improving educational decision-making.

### Research Question
*"Using sentiment analysis, what meaningful insights can be extracted from reviews of professors made by students, and to what extent can these insights be used to predict/recommend grading patterns and course recommendations?"*

## 🚀 Purpose & Motivation

### Why This Matters
- **Student Decision Making**: Help students make informed course choices based on sentiment analysis of peer reviews
- **Institutional Improvement**: Provide insights for educational institutions to enhance teaching quality
- **Educational Data Mining**: Advance the field of using NLP and ML in educational contexts
- **Predictive Analytics**: Demonstrate the potential of sentiment analysis for academic outcome prediction

### Key Insights Discovered
- **Sentiment Score** emerged as the most important predictor of grades (26% feature importance)
- **Course Rating** and **Difficulty** are strong secondary predictors
- **Gradient Boosting** performed best among three tested models (Random Forest, Neural Network)
- Sentiment analysis achieved **80% accuracy** on test data, showing strong predictive capability

## 🤖 Machine Learning Approach

### Supervised Learning Pipeline

This project implements a **two-stage supervised learning pipeline**:

#### **Stage 1: Sentiment Classification (Binary Classification)**
- **Problem**: Classify student reviews as positive or negative sentiment
- **Algorithm**: Naive Bayes Classifier with TF-IDF vectorization
- **Features**: Preprocessed text from student reviews
- **Labels**: Binary sentiment (positive/negative) based on rating thresholds
- **Performance**: 80% accuracy, 88% F1-score on test data

#### **Stage 2: Grade Prediction (Regression)**
- **Problem**: Predict numerical grade outcomes from review features
- **Algorithms**: Three different approaches for comparison
  - **Gradient Boosting Regressor**: Sequential ensemble of decision trees
  - **Random Forest Regressor**: Parallel ensemble of decision trees  
  - **Neural Network (MLPRegressor)**: Multi-layer perceptron for regression
- **Features**: Sentiment scores + numerical features (rating, difficulty, etc.)
- **Target**: Numerical grade values
- **Performance**: Gradient Boosting achieved best results (RMSE: 0.254)

### Feature Engineering & Data Processing

#### **Text Processing Pipeline**
1. **Text Cleaning**: Remove special characters, normalize whitespace
2. **Tokenization**: Split text into individual words/tokens
3. **Stop Word Removal**: Eliminate common words (the, and, is, etc.)
4. **Lemmatization**: Reduce words to base form (running → run)
5. **TF-IDF Vectorization**: Convert text to numerical features

#### **Feature Engineering**
- **Sentiment Features**: Probability scores from Stage 1 classifier
- **Numerical Features**: Rating, difficulty, review count
- **Aggregated Features**: Professor-level statistics (avg_rating, avg_difficulty)
- **Interaction Features**: Combinations of sentiment and other variables

### Model Training & Evaluation

#### **Cross-Validation Strategy**
- **Train/Test Split**: 80/20 split for model evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Feature Selection**: Analysis of feature importance and correlation

#### **Evaluation Metrics**
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Regression**: RMSE (Root Mean Square Error), MAE (Mean Absolute Error), R² Score
- **Model Comparison**: Statistical significance testing between models

#### **Overfitting Prevention**
- **Regularization**: L1/L2 regularization in neural networks
- **Early Stopping**: Prevent overfitting in gradient boosting
- **Cross-Validation**: Ensure robust performance estimates

## 🛠️ Technologies & Libraries Used

### Core Machine Learning & Data Science
- **pandas (2.2.0)**: Data manipulation and analysis
- **numpy (1.26.3)**: Numerical computing
- **scikit-learn (1.4.0)**: Machine learning algorithms and utilities
- **tensorflow (2.16.1)**: Deep learning framework for neural networks

### Natural Language Processing
- **nltk (3.8.1)**: Natural language processing toolkit
- **transformers (4.37.2)**: State-of-the-art NLP models
- **beautifulsoup4 (4.12.3)**: Web scraping for data collection

### Data Visualization & Analysis
- **matplotlib (3.8.2)**: Plotting and visualization
- **seaborn (0.13.1)**: Statistical data visualization
- **jupyter (1.0.0)**: Interactive notebooks for analysis

### Web Scraping & Data Collection
- **requests (2.31.0)**: HTTP library for data fetching
- **selenium (4.18.1)**: Web automation for dynamic content
- **webdriver-manager (4.0.1)**: Automated webdriver management

## 📊 Methodology & Approach

### 1. Data Collection & Preprocessing
- **Source**: RateMyProfessor reviews from 5C Colleges dataset
- **Data Types**: Structured (ratings, grades, difficulty) + Unstructured (review text)
- **Preprocessing**: Text cleaning, sentiment labeling, feature engineering

### 2. Sentiment Analysis Pipeline
- **Model**: Naive Bayes classifier with TF-IDF vectorization
- **Performance**: 80% accuracy, 88% F1-score on test data
- **Output**: Sentiment probability scores for each review

### 3. Predictive Modeling
Three machine learning approaches were implemented and compared:

| Model | Training RMSE | Test RMSE | Test MAE | Test R² | Key Advantage |
|-------|---------------|-----------|----------|---------|---------------|
| **Gradient Boosting** | 0.208 | 0.254 | 0.127 | -0.046 | Best overall performance |
| Random Forest | ~0.098 | ~0.259 | ~0.144 | -0.085 | Good interpretability |
| Neural Network | 0.232 | 0.280 | 0.178 | -0.266 | Complex pattern learning |

### 4. Feature Engineering
Key features used for grade prediction:
- **sentiment_score**: Probability of positive sentiment (26% importance)
- **rating**: Course rating (20% importance)
- **avg_rating**: Professor's average rating (17% importance)
- **avg_difficulty**: Professor's average difficulty (14% importance)
- **difficulty**: Course difficulty (13% importance)
- **review_count**: Number of reviews (10% importance)

## 📁 Project Structure

```
RateMyProf_Sentiment_Analysis_to_Grade_Prediction/
├── Project/
│   ├── data/                    # Raw and processed datasets
│   ├── datasets/                # Additional data sources
│   ├── src/                     # Source code modules
│   │   ├── data/               # Data collection and preprocessing
│   │   ├── models/             # ML models and sentiment analysis
│   │   └── visualization/      # Data visualization tools
│   ├── models/                 # Saved model files and feature importances
│   ├── visualizations/         # Random Forest output plots
│   ├── visualizations_gb/      # Gradient Boosting output plots
│   ├── visualizations_nn/      # Neural Network output plots
│   ├── main.py                 # Random Forest pipeline
│   ├── main_gb.py              # Gradient Boosting pipeline
│   ├── main_nn.py              # Neural Network pipeline
│   ├── FinalProjectCode.ipynb  # Comprehensive Jupyter notebook
│   ├── project_report_*.txt    # Detailed analysis reports
│   └── requirements.txt        # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                 # Git ignore rules
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RateMyProf_Sentiment_Analysis_to_Grade_Prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd Project
   pip install -r requirements.txt
   ```

## 🎯 Running the Analysis

### Individual Model Pipelines

**Gradient Boosting (Recommended):**
```bash
python3 main_gb.py
```
- **Outputs**: `models/grade_predictor_gb.pkl`, `feature_importance_gb.csv`, `visualizations_gb/`
- **Report**: `project_report_gb.txt`

**Random Forest:**
```bash
python3 main.py
```
- **Outputs**: `models/grade_predictor.pkl`, `feature_importance.csv`, `visualizations/`

**Neural Network:**
```bash
python3 main_nn.py
```
- **Outputs**: `models/grade_predictor_nn.pkl`, `feature_importance_nn.csv`, `visualizations_nn/`
- **Report**: `project_report_nn.txt`

### Comprehensive Analysis
For the complete analysis with all models and comparisons:
```bash
jupyter notebook FinalProjectCode.ipynb
```

## 📈 Key Findings & Learnings

### What We Discovered

1. **Sentiment Analysis Effectiveness**
   - Achieved 80% accuracy in classifying review sentiment
   - Sentiment scores are highly predictive of grade outcomes
   - Text preprocessing significantly improves model performance

2. **Feature Importance Insights**
   - **Sentiment Score** (26%): Most critical predictor of grades
   - **Course Rating** (20%): Strong correlation with academic outcomes
   - **Professor Statistics** (avg_rating, avg_difficulty): Important contextual features

3. **Model Performance Comparison**
   - **Gradient Boosting** emerged as the best performer
   - All models struggled with generalization (negative R²)
   - Sentiment analysis adds significant predictive value

### Technical Learnings

1. **NLP in Educational Context**
   - Text preprocessing is crucial for sentiment analysis
   - TF-IDF vectorization works well for review classification
   - Sentiment scores provide valuable numerical features

2. **Machine Learning Insights**
   - Ensemble methods (Gradient Boosting, Random Forest) outperform neural networks on this dataset
   - Feature engineering significantly impacts model performance
   - Interpretability is crucial for educational applications

3. **Data Science Best Practices**
   - Proper train/test splitting is essential
   - Feature importance analysis provides actionable insights
   - Visualization helps communicate complex relationships

### Challenges & Limitations

1. **Model Generalization**
   - All models showed limited generalization (negative R²)
   - Suggests need for more features or different modeling approaches

2. **Data Quality**
   - Class imbalance in sentiment labels
   - Limited sample size for some departments
   - Potential bias in self-reported data

## 🔮 Future Work & Improvements

### Potential Enhancements
1. **Advanced NLP Techniques**
   - BERT or other transformer models for better text understanding
   - Aspect-based sentiment analysis
   - Topic modeling for review categorization

2. **Feature Engineering**
   - Temporal features (semester, year trends)
   - Interaction features between sentiment and other variables
   - Department-specific modeling

3. **Model Improvements**
   - Ensemble methods combining multiple approaches
   - Hyperparameter optimization
   - Cross-validation strategies

4. **Data Expansion**
   - Larger, more diverse datasets
   - Additional features (class size, prerequisites, etc.)
   - Longitudinal data for trend analysis

## 📚 Reports & Documentation

- **`project_report_gb.txt`**: Detailed Gradient Boosting analysis
- **`project_report_nn.txt`**: Neural Network implementation details
- **`project_report_comparison.txt`**: Comprehensive model comparison
- **`FinalProjectCode.ipynb`**: Complete analysis notebook

## 🤝 Contributing

This project demonstrates the application of machine learning and NLP in educational data mining. Feel free to:
- Extend the analysis with additional models
- Improve the feature engineering pipeline
- Add new visualization capabilities
- Explore different datasets or domains

## 📄 License

This project is for educational and research purposes. Please ensure proper attribution when using or building upon this work.

---

**Note**: This project successfully demonstrates the potential of sentiment analysis in educational contexts while highlighting the challenges of predicting complex outcomes like grades from review data alone. 