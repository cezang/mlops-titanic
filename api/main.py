from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pandas import DataFrame
from classification_model.predict import make_prediction 
from classification_model.processing.validation import MultipleDataInputs, PredictionResult
import logging
from pathlib import Path
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development only; restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Serwuj statyczne pliki z katalogu 'static'
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Dodaj endpoint dla głównej strony
@app.get("/")
async def read_root():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)

@app.post("/predict", response_model=PredictionResult)
async def predict(input_data: MultipleDataInputs):
    """
    Przyjmuje listę danych wejściowych w formacie JSON, wykonuje predykcję
    i zwraca wyniki.
    """
    try:
        data_dicts = [item.dict() for item in input_data.inputs] 
        data = DataFrame(data_dicts)
        logging.info(f"Reading data: {data.head()}")
        data.rename(columns={"home_dest": "home.dest"}, inplace=True)
        prediction_result = make_prediction(input_data=data)
        logging.info(f"Prediction result: {prediction_result}")
        
        if prediction_result["errors"]:
            logging.warning(f"Errors encountered: {prediction_result['errors']}")
            raise HTTPException(
                status_code=400, detail=prediction_result["errors"]
            )
        
        return prediction_result

    except Exception as e:
        print("Error occurred:", str(e))
        raise HTTPException(status_code=500, detail=str(e))
