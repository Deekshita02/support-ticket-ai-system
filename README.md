# Support Ticket AI System

An AI-powered system for querying support ticket data in natural language and detecting anomalies, built with FastAPI, Streamlit, and an LLM (via Groq's free tier).

## Architecture Overview

- **Data layer**: CSV loaded into a pandas DataFrame at startup (`support_tickets.csv`, 500 rows).
- **NL query layer** (`llm_query.py`): The LLM does NOT answer questions directly from raw data (to avoid hallucinated numbers). Instead, it translates a natural language question into a structured JSON "query spec" (operation, filters, target column, group-by). This spec is validated against an allow-list of columns/operations/operators, then executed deterministically in pandas. This guarantees numeric accuracy — the LLM only handles intent parsing, never arithmetic.
- **Anomaly detection** (`anomaly.py`): Pure rule-based logic, no LLM involved (deliberate choice — anomaly rules are deterministic and don't need NL understanding):
  - Abnormally long resolution times: `resolution_time_hrs > mean + 2*std` (Resolved tickets only)
  - Unresolved high-priority tickets older than a configurable threshold (default 24h), using the latest timestamp in the dataset as the reference "now" for reproducibility.
- **API layer** (`main.py`): FastAPI with 3 endpoints (see below).
- **UI layer** (`ui.py`): Streamlit app with two tabs — ask a question, or run anomaly detection with an adjustable time threshold.

## Model / Tools Used

- **LLM**: `openai/gpt-oss-120b` via **Groq's free-tier API** (OpenAI-compatible endpoint). Chosen for strong instruction-following (needed for reliable structured JSON output) and Groq's fast, zero-cost inference.
- **Backend**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Data**: pandas

## Handling of Missing Data

The dataset's only nulls are in `resolution_time_hrs` and `customer_rating`, and they occur exactly for tickets with `status` in `{Open, Escalated}` — i.e., tickets that haven't been resolved yet. These are not corrupted/missing values; they are structurally absent. No imputation is performed. Instead:
- Aggregations (average/sum/min/max) automatically drop nulls via `.dropna()`.
- Anomaly detection explicitly filters to resolved tickets before computing resolution-time statistics.
- The null pattern itself is used as a signal (e.g., "unresolved" = `status != Resolved`).

## Setup Instructions

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key_here"
```

Place `support_tickets.csv` in the project root (or set `DATA_PATH` env var to its location).

## Running the System

**Start the API:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Start the UI (in a separate terminal):**
```bash
streamlit run ui.py
```

The UI expects the API to be running at `http://localhost:8000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check, returns row count loaded |
| POST | `/query` | Body: `{"question": "..."}`. Returns spec + computed result |
| GET | `/anomalies?hours=24` | Returns detected anomalies, threshold configurable |

## Example Queries & Outputs

**Query:** "How many critical tickets are unresolved?"
```json
{
  "question": "How many critical tickets are unresolved?",
  "spec": {"operation": "count", "filters": [{"column": "priority", "op": "==", "value": "Critical"}, {"column": "status", "op": "!=", "value": "Resolved"}]},
  "result": {"result": 31}
}
```

**Query:** "What is the average customer rating for Technical category tickets?"
```json
{"result": {"result": 3.74}}
```

**GET /anomalies:**
```json
{"total_anomalies": 97, "anomalies": [{"ticket_id": "TKT-023", "anomaly_type": "abnormally_long_resolution", ...}]}
```

## Known Limitations

- Time-based phrasing (e.g., "older than 12 hours") in NL queries is not parsed into a time filter by the LLM layer; this specific logic is instead handled deterministically in the anomaly detection endpoint, since reasoning about "current time" from an LLM is unreliable and the dataset is historical (not live).
- The LLM occasionally may propose a filter on a column/operator outside the allow-list; this is caught by `validate_spec()` and returned as a clear error rather than crashing or producing an incorrect result.
- No authentication on API endpoints (out of scope for this assessment).
- Groq free-tier has daily/rate limits; heavy concurrent use may hit them.

## What I'd Improve With More Time

- Support compound time-based NL queries (e.g. "critical tickets open more than 12 hours") end-to-end through the LLM layer.
- Add caching for repeated identical questions.
- Add unit tests for `validate_spec` and `execute_spec` edge cases.
- Containerize with Docker for guaranteed single-command startup across environments.
