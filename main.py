import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import llm_query
import anomaly

app = FastAPI(title="Support Ticket AI System")

DATA_PATH = os.environ.get("DATA_PATH", "support_tickets.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


df = load_data()


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": int(len(df))}


@app.post("/query")
def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    try:
        return llm_query.answer_question(df, request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


@app.get("/anomalies")
def anomalies(hours: int = 24):
    try:
        return anomaly.detect_anomalies(df, unresolved_hours=hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect anomalies: {str(e)}")
