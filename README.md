# RateMyProf Grade Predictor 🎓

An AI-powered web application that predicts your grade based on professor reviews and your personal study habits.

![Project Banner](https://img.shields.io/badge/Status-Complete-green) ![Tech](https://img.shields.io/badge/Stack-Next.js%20%7C%20FastAPI%20%7C%20BERT-blue)

## Features

- **Multi-Model Prediction**: Combines sentiment analysis of professor reviews with student habit modeling.
- **BERT Embeddings**: Uses `all-MiniLM-L6-v2` for state-of-the-art text understanding.
- **Personalized**: Factors in your motivation, study hours, and prior GPA.
- **Modern UI**: Built with Next.js, Tailwind CSS, and Framer Motion for smooth animations.

---

## 🚀 Quick Start

### 1. Backend API (Python)

```bash
cd Project
source venv/bin/activate
python api/main.py
```
*Server running at http://localhost:8000*

### 2. Frontend (Next.js)

Open a new terminal:
```bash
cd Project/frontend
npm run dev
```
*App running at http://localhost:3000*

---

## Project Structure

```
Project/
├── api/                  # FastAPI Backend
│   └── main.py
├── frontend/             # Next.js Frontend
│   ├── src/app/          # Pages & Layouts
│   └── src/components/   # UI Components
├── src/                  # ML Source Code
│   ├── models/           # Training logic
│   └── data/             # Data processing
└── models/               # Saved PKL models
```

## Tech Stack

- **Frontend**: Next.js 14, React, Tailwind CSS, Framer Motion, Radix UI
- **Backend**: FastAPI, Uvicorn
- **ML**: Scikit-learn, Sentence-Transformers (BERT), Pandas
- **Data**: RateMyProfessor reviews + 80k Student Habits dataset

## Algorithms

1. **Sentiment Analysis**: BERT-encoded text features trained on review sentiment.
2. **Grade Prediction**: Gradient Boosting Regressor tuned via GridSearchCV.
3. **Habits Model**: Separate model trained on student behavior data (R²=0.87).
4. **Ensemble**: Weighted average of review-based and habits-based predictions, with context-aware scaling for class difficulty.

## 🚀 Deployment

### Option 1: Vercel (Frontend & Backend)
1. Fork this repository.
2. In Vercel, import the project.
3. Set **Root Directory** to `Project/frontend`.
4. Deploy!

*Note: For the Python API to run on Vercel, you'll need to configure `api/index.py` as a serverless function entry point or deploy the backend separately to Railway/Render.*

### Option 2: Railway (Recommended for API)
1. Deploy the `Project/` folder to Railway.
2. Set Build Command: `pip install -r requirements.txt`.
3. Set Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.

## 📂 Data Sources
- **RateMyProfessors**: Scraped review data (simulated/cached).
- **Student Habits**: Synthetic dataset of 80,000 student records including GPA, study hours, and motivation.