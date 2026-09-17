import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Support Ticket AI", layout="wide")
st.title("Support Ticket AI System")

tab1, tab2 = st.tabs(["Ask a Question", "Anomaly Detection"])

with tab1:
    st.subheader("Ask a natural language question about the tickets")
    question = st.text_input("Your question", placeholder="e.g. How many critical tickets are unresolved?")

    if st.button("Ask", key="ask_btn"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=30)
                    data = resp.json()
                    if resp.status_code != 200:
                        st.error(data.get("detail", "Something went wrong."))
                    else:
                        result = data.get("result", {})
                        if "error" in result:
                            st.error(result["error"])
                        elif "result" in result and isinstance(result["result"], list):
                            st.write(f"Matched {result.get('matched_total', len(result['result']))} tickets. Showing top results:")
                            st.dataframe(pd.DataFrame(result["result"]))
                        elif "result" in result and isinstance(result["result"], dict):
                            st.write("Result (grouped):")
                            st.bar_chart(pd.Series(result["result"]))
                        else:
                            st.success(f"Answer: {result.get('result')}")
                        with st.expander("Debug: query spec used"):
                            st.json(data.get("spec"))
                except Exception as e:
                    st.error(f"Request failed: {e}")

    st.caption("Example questions: 'How many tickets are currently open?', 'Which agent has the most tickets?', 'Average customer rating for Technical tickets'")

with tab2:
    st.subheader("Detect anomalies in ticket data")
    hours = st.slider("Unresolved age threshold (hours)", min_value=1, max_value=72, value=24)

    if st.button("Run anomaly detection", key="anomaly_btn"):
        with st.spinner("Scanning for anomalies..."):
            try:
                resp = requests.get(f"{API_URL}/anomalies", params={"hours": hours}, timeout=30)
                data = resp.json()
                st.metric("Total anomalies found", data["total_anomalies"])
                if data["anomalies"]:
                    st.dataframe(pd.DataFrame(data["anomalies"]))
                else:
                    st.info("No anomalies found.")
            except Exception as e:
                st.error(f"Request failed: {e}")
