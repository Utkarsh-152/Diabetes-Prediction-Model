from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.ml_model.pipelines.prediction_pipeline import PredictPipeline, CustomData
from typing import Optional

app = FastAPI(title="Diabetes Prediction API",
              description="API for predicting diabetes risk using machine learning",
              version="1.0.0")

# Set up templates folder
templates = Jinja2Templates(directory="templates")

# Optional: Serve static files if needed (e.g., CSS, JS, images)
# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    """Render the prediction form page"""
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_result(
    request: Request,
    HighBP: str = Form(...),
    HighChol: str = Form(...),
    BMI: float = Form(...),
    Stroke: str = Form(...),
    HeartDiseaseorAttack: str = Form(...),
    HvyAlcoholConsump: str = Form(...),
    GenHlth: str = Form(...),
    MentHlth: int = Form(...),
    PhysHlth: int = Form(...),
    DiffWalk: str = Form(...),
    Age: int = Form(...),
    Income: int = Form(...),
):
    """Process the form submission and show prediction results"""

    # Convert string values to integers for the model
    high_bp_value = 1 if HighBP == "yes" else 0
    high_chol_value = 1 if HighChol == "yes" else 0
    stroke_value = 1 if Stroke == "yes" else 0
    heart_disease_value = 1 if HeartDiseaseorAttack == "yes" else 0
    alcohol_value = 1 if HvyAlcoholConsump == "yes" else 0
    diff_walk_value = 1 if DiffWalk == "yes" else 0

    # Convert general health to numeric scale (1-5)
    gen_hlth_map = {
        "excellent": 1,
        "very good": 2,
        "good": 3,
        "fair": 4,
        "poor": 5
    }
    gen_hlth_value = gen_hlth_map.get(GenHlth, 3)  # Default to 'good' if not found

    # For now, just return the form data (in a real app, you would use the prediction pipeline)
    # In a future implementation, you would use:
    data = CustomData(high_bp_value, high_chol_value, BMI, stroke_value, heart_disease_value,
                      alcohol_value, gen_hlth_value, MentHlth, PhysHlth, diff_walk_value, Age, Income)
    df = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    # For now, simulate a prediction (0 = no diabetes, 1 = diabetes)
    # This is just a placeholder - in a real app you would use the actual model
    #prediction = 1 if (high_bp_value + high_chol_value + (BMI > 30) + stroke_value + heart_disease_value) >= 2 else 0

    # Prepare risk factors to display
    risk_factors = []
    if high_bp_value == 1:
        risk_factors.append("High Blood Pressure")
    if high_chol_value == 1:
        risk_factors.append("High Cholesterol")
    if BMI > 30:
        risk_factors.append(f"BMI above 30 (Your BMI: {BMI:.1f})")
    if stroke_value == 1:
        risk_factors.append("History of Stroke")
    if heart_disease_value == 1:
        risk_factors.append("Heart Disease")
    if alcohol_value == 1:
        risk_factors.append("Heavy Alcohol Consumption")
    if gen_hlth_value >= 4:
        risk_factors.append("Poor General Health")
    if MentHlth > 14:
        risk_factors.append("Poor Mental Health")
    if PhysHlth > 14:
        risk_factors.append("Poor Physical Health")
    if diff_walk_value == 1:
        risk_factors.append("Difficulty Walking")

    # Return the results page with the prediction and form data
    return templates.TemplateResponse(
        "predict.html",
        {
            "request": request,
            "prediction": prediction,
            "risk_factors": risk_factors,
            "show_results": True,
            "form_data": {
                "HighBP": HighBP,
                "HighChol": HighChol,
                "BMI": BMI,
                "Stroke": Stroke,
                "HeartDiseaseorAttack": HeartDiseaseorAttack,
                "HvyAlcoholConsump": HvyAlcoholConsump,
                "GenHlth": GenHlth,
                "MentHlth": MentHlth,
                "PhysHlth": PhysHlth,
                "DiffWalk": DiffWalk,
                "Age": Age,
                "Income": Income
            }
        }
    )

@app.get("/model-evaluation", response_class=HTMLResponse)
async def model_evaluation_page(request: Request):
    """Render the model evaluation page"""
    return templates.TemplateResponse("model_evaluation.html", {"request": request})

