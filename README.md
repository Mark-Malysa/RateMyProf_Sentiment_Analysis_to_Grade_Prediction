# Analysis of Student Reviews on Professors with Predictive Modeling

## Project Overview
This project analyzes student reviews of professors using sentiment analysis and predictive modeling to extract meaningful insights about professor performance and student outcomes. Multiple machine learning models are compared for grade prediction.

## Research Question
Using sentiment analysis, what meaningful insights can be extracted from reviews of professors made by students, and to what extent can these insights be used to predict/recommend grading patterns and course recommendations?

## Project Structure
```
.
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── data/          # Data collection and preprocessing
│   ├── models/        # Machine learning models
│   └── visualization/ # Visualization code
├── models/            # Saved model files and feature importances
├── visualizations/    # Output plots for Random Forest
├── visualizations_gb/ # Output plots for Gradient Boosting
├── visualizations_nn/ # Output plots for Neural Network
├── main.py            # Random Forest pipeline
├── main_gb.py         # Gradient Boosting pipeline
├── main_nn.py         # Neural Network pipeline
├── project_report_gb.txt         # Gradient Boosting report
├── project_report_nn.txt         # Neural Network report
├── project_report_comparison.txt # Model comparison report
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Data Sources
1. Rate My Professor Reviews 5C Colleges Kaggle Dataset

## Methodology
1. Data Collection and Preprocessing
   - Web scraping from RateMyProfessor
   - Data cleaning and standardization
   - Text preprocessing and sentiment analysis

2. Analysis Techniques
   - Sentiment Analysis using NLP models
   - Trend and Correlation Analysis
   - Predictive Modeling for student performance (Random Forest, Gradient Boosting, Neural Network)

3. Tools and Libraries
   - Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn)
   - NLP tools (NLTK, scikit-learn)
   - Web scraping (BeautifulSoup)

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipelines
You can run each model pipeline independently:

- **Random Forest:**
  ```bash
  python3 main.py
  ```
  - Outputs: models/grade_predictor.pkl, feature_importance.csv, visualizations/

- **Gradient Boosting:**
  ```bash
  python3 main_gb.py
  ```
  - Outputs: models/grade_predictor_gb.pkl, feature_importance_gb.csv, visualizations_gb/
  - Report: project_report_gb.txt

- **Neural Network:**
  ```bash
  python3 main_nn.py
  ```
  - Outputs: models/grade_predictor_nn.pkl, feature_importance_nn.csv, visualizations_nn/
  - Report: project_report_nn.txt

## Reports
- **project_report_gb.txt:** Detailed report on the Gradient Boosting model, including methodology, results, and feature importance.
- **project_report_nn.txt:** Detailed report on the Neural Network model, including methodology, results, and feature importance.
- **project_report_comparison.txt:** Comparative analysis of Random Forest, Gradient Boosting, and Neural Network models, discussing performance, interpretability, and practical implications.

## Project Status
- [x] Data Collection
- [x] Data Preprocessing
- [x] Sentiment Analysis
- [x] Predictive Modeling (Random Forest, Gradient Boosting, Neural Network)
- [x] Visualization and Reporting
- [x] Model Comparison and Documentation

## Notes
- All scripts use the same data pipeline for fair comparison.
- Visualizations and feature importances are saved in separate directories for each model.
- For further analysis or to add new models, extend the corresponding main script and update the reports accordingly. 