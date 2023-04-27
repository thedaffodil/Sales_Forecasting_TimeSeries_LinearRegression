from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version

app = FastAPI()

class Predicting(BaseModel):
    date1: int
    date2: int



app = FastAPI()


# prediction function
def make_sales_prediction(request):
    # parse input from request
    date1 = request['date1']
    date2 = request['date2']
    

    # Make an input vector
    dates = [[date1,date2]]

    # Predict
    prediction = predict_pipeline(dates)

    return prediction

# Prediction endpoint
@app.post("/predict")
def predict_sales(request: Predicting):
    prediction = make_sales_prediction(request.dict())
    return prediction