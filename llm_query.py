import os
import json
import operator
from openai import OpenAI

SCHEMA_DESCRIPTION = """
Columns available in the dataset:
- ticket_id (string)
- created_at (datetime)
- category (Billing | Technical | General)
- priority (Low | Medium | High | Critical)
- status (Open | Resolved | Escalated)
- response_time_hrs (float)
- resolution_time_hrs (float, null if not resolved)
- agent_id (string, e.g. AGT-04)
- customer_rating (integer 1-5, null if not resolved)
- issue_summary (free text)
"""

SYSTEM_PROMPT = f"""You convert a user's question about a support ticket dataset into a JSON query spec.
{SCHEMA_DESCRIPTION}

Return ONLY valid JSON (no markdown, no explanation) matching this shape:
{{
  "operation": "count" | "average" | "sum" | "min" | "max" | "list" | "group_count",
  "target_column": "<column name or null>",
  "filters": [
     {{"column": "<column>", "op": "==" | "!=" | ">" | "<" | ">=" | "<=" | "contains" | "is_null" | "not_null", "value": "<value or null>"}}
  ],
  "group_by": "<column name or null>",
  "limit": <int or null>
}}

Rules:
- "operation" = "count" for questions like "how many...".
- "operation" = "average"/"sum"/"min"/"max" requires a numeric "target_column".
- "operation" = "group_count" when the question asks "which agent has the most/least..." — set "group_by" to that column.
- "operation" = "list" when the user wants to see actual matching tickets — set "limit" (default 10 if unspecified).
- For "unresolved" tickets, use column "status", op "!=", value "Resolved".
- For time-based questions like "older than 24 hours", skip time filters — leave that reasoning to the code layer.
- Only output the JSON object, nothing else.
"""

ALLOWED_COLUMNS = {
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary",
}
ALLOWED_OPERATIONS = {"count", "average", "sum", "min", "max", "group_count", "list"}
ALLOWED_FILTER_OPS = {"==", "!=", ">", "<", ">=", "<=", "is_null", "not_null", "contains"}

OPS = {
    "==": operator.eq, "!=": operator.ne,
    ">": operator.gt, "<": operator.lt,
    ">=": operator.ge, "<=": operator.le,
}

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


def get_query_spec(question: str) -> dict:
    client = get_client()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"operation": "invalid", "raw_response": raw}


def validate_spec(spec: dict):
    if not isinstance(spec, dict):
        return False, "Query specification must be a JSON object."
    operation = spec.get("operation")
    if operation not in ALLOWED_OPERATIONS:
        return False, f"Unsupported operation: {operation}"
    target_column = spec.get("target_column")
    if target_column is not None and target_column not in ALLOWED_COLUMNS:
        return False, f"Invalid target_column: {target_column}"
    group_by = spec.get("group_by")
    if group_by is not None and group_by not in ALLOWED_COLUMNS:
        return False, f"Invalid group_by column: {group_by}"
    filters = spec.get("filters") or []
    if not isinstance(filters, list):
        return False, "filters must be a list."
    for f in filters:
        if not isinstance(f, dict):
            return False, "Each filter must be an object."
        if f.get("column") not in ALLOWED_COLUMNS:
            return False, f"Invalid filter column: {f.get('column')}"
        if f.get("op") not in ALLOWED_FILTER_OPS:
            return False, f"Invalid filter operator: {f.get('op')}"
    return True, None


def apply_filters(data, filters):
    result = data
    for f in filters:
        col, op, val = f["column"], f["op"], f.get("value")
        if col not in result.columns:
            continue
        if op == "is_null":
            result = result[result[col].isna()]
        elif op == "not_null":
            result = result[result[col].notna()]
        elif op == "contains":
            result = result[result[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op in OPS:
            try:
                val_cast = float(val)
                result = result[OPS[op](result[col], val_cast)]
            except (ValueError, TypeError):
                result = result[OPS[op](result[col], val)]
    return result


def execute_spec(data, spec: dict):
    op = spec.get("operation")
    filters = spec.get("filters") or []
    target = spec.get("target_column")
    group_by = spec.get("group_by")
    limit = spec.get("limit") or 10

    filtered = apply_filters(data, filters)

    if op == "count":
        return {"result": int(len(filtered))}
    elif op in ("average", "sum", "min", "max"):
        if not target or target not in filtered.columns:
            return {"error": f"Invalid target_column: {target}"}
        series = filtered[target].dropna()
        if series.empty:
            return {"result": None, "note": "No matching rows with non-null values"}
        val = getattr(series, {"average": "mean", "sum": "sum", "min": "min", "max": "max"}[op])()
        return {"result": round(float(val), 2)}
    elif op == "group_count":
        if not group_by or group_by not in filtered.columns:
            return {"error": f"Invalid group_by: {group_by}"}
        counts = filtered.groupby(group_by).size().sort_values(ascending=False)
        return {"result": counts.to_dict()}
    elif op == "list":
        cols = ["ticket_id", "category", "priority", "status", "agent_id", "issue_summary"]
        rows = filtered[cols].head(limit).to_dict(orient="records")
        return {"result": rows, "matched_total": int(len(filtered))}
    else:
        return {"error": f"Unsupported operation: {op}"}


def answer_question(data, question: str) -> dict:
    spec = get_query_spec(question)
    is_valid, error = validate_spec(spec)
    if not is_valid:
        return {"question": question, "spec": spec, "result": {"error": error}}
    result = execute_spec(data, spec)
    return {"question": question, "spec": spec, "result": result}
