import os
import logging
import argparse
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from model import Model
import uvicorn
from datetime import datetime

# --- Создание папки tmp ---
os.makedirs("tmp", exist_ok=True)

# --- Инициализация приложения ---
app = FastAPI()
app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

# --- Логирование ---
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S IST")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

# --- Инициализация модели ---
try:
    model = Model(threshold=0.3)  # Замените на ваш порог
    app_logger.info("Model initialized successfully")
except Exception as e:
    app_logger.error(f"Failed to initialize model: {str(e)}")
    raise

# --- Эндпоинты ---

@app.get("/health")
def health():
    return {"status": "OK", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("start_form.html", {"request": request})

@app.post("/process")
async def process_request(file: UploadFile, request: Request):
    save_pth = f"tmp/{file.filename}"
    app_logger.info(f"Processing file: {save_pth}")
    try:
        content = await file.read()
        with open(save_pth, "wb") as fid:
            fid.write(content)
        app_logger.info(f"File saved: {save_pth}, size: {len(content)} bytes")
        
        # Чтение CSV
        df = pd.read_csv(save_pth, index_col=None)  # Без index_col=0 для гибкости
        app_logger.info(f"CSV loaded: shape {df.shape}, columns: {df.columns.tolist()}")
        
        # Предсказания
        predictions = model(df)
        results = [{"id": int(id_), "prediction": int(pred)} for id_, pred in predictions]
        submission_df = pd.DataFrame(results)
        submission_df.to_csv("submission.csv", index=False)
        app_logger.info(f"Predictions saved to submission.csv with {len(results)} entries")
        return {"predictions": results, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}
    except Exception as e:
        app_logger.error(f"Full error processing file: {str(e)}")
        import traceback
        app_logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}, status_code=500)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile):
    save_path = f"tmp/{file.filename}"
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        df = pd.read_csv(save_path, index_col=None)
        predictions = model(df)
        results = [{"id": int(id_), "prediction": int(pred)} for id_, pred in predictions]
        pd.DataFrame(results).to_csv("submission.csv", index=False)
        return {"predictions": results, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}
    except Exception as e:
        return JSONResponse(content={"error": str(e), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}, status_code=500)

# --- Запуск приложения ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    app_logger.info(f"Starting application on {args['host']}:{args['port']} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    uvicorn.run("app:app", host=args['host'], port=args['port'], reload=True)