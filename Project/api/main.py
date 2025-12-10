"""
FastAPI Backend for Grade Prediction Web App.

This API serves the ML models for the Next.js frontend.
Endpoints:
- POST /api/predict - Combined grade prediction

Author: Enhanced for Resume Portfolio
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add parent directory to path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Grade Prediction API",
    description="ML-powered grade prediction based on professor reviews and student habits",
    version="1.0.0"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request body for grade prediction."""
    rating: float = Field(..., ge=1, le=5, description="Professor rating (1-5)")
    difficulty: float = Field(..., ge=1, le=5, description="Course difficulty (1-5)")
    review_text: Optional[str] = Field("", description="Professor review text")
    study_hours: float = Field(..., ge=0, le=12, description="Daily study hours")
    prior_gpa: float = Field(..., ge=0, le=4, description="Previous GPA (0-4.0)")
    motivation: int = Field(..., ge=1, le=10, description="Motivation level (1-10)")


class PredictionResponse(BaseModel):
    """Response body for grade prediction."""
    combined_gpa: float
    combined_grade: str
    recommendation: str
    breakdown: dict
    success: bool = True


# Global model instances (loaded once at startup)
grade_predictor = None
habits_model = None


@app.on_event("startup")
async def load_models():
    """Load ML models on startup."""
    global grade_predictor, habits_model
    
    try:
        from src.models.enhanced_grade_predictor import EnhancedGradePredictor
        from src.models.student_habits_model import StudentHabitsModel
        
        # Load grade predictor
        grade_predictor = EnhancedGradePredictor(use_bert=True, use_hyperparameter_tuning=False)
        grade_predictor.load_model('models/grade_predictor_enhanced.pkl')
        
        # Load habits model
        habits_model = StudentHabitsModel()
        habits_model.load_model('models/student_habits_model.pkl')
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("API will still run but predictions may fail.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "Grade Prediction API is running",
        "models_loaded": grade_predictor is not None
    }


@app.get("/api/health")
async def health():
    """Health check for deployment."""
    return {"status": "healthy"}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_grade(request: PredictionRequest):
    """
    Predict grade based on professor review and student habits.
    
    Combines:
    - Professor sentiment analysis (from review text and ratings)
    - Student habit factors (study hours, prior GPA, motivation)
    
    Returns personalized grade prediction with recommendation.
    """
    if grade_predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Use the combined prediction method
        result = grade_predictor.combined_predict(
            review_text=request.review_text or f"Rating: {request.rating}, Difficulty: {request.difficulty}",
            rating=request.rating,
            difficulty=request.difficulty,
            study_hours=request.study_hours,
            prior_gpa=request.prior_gpa,
            motivation=request.motivation,
            review_weight=0.5  # 50/50 split as discussed
        )
        
        return PredictionResponse(
            combined_gpa=result['combined_gpa'],
            combined_grade=result['combined_grade'],
            recommendation=result['recommendation'],
            breakdown=result['breakdown'],
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-habits")
async def predict_from_habits(study_hours: float, prior_gpa: float, motivation: int):
    """Predict grade from habits only (utility endpoint)."""
    if habits_model is None:
        raise HTTPException(status_code=503, detail="Habits model not loaded")
    
    result = habits_model.predict_gpa(study_hours, prior_gpa, motivation)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
