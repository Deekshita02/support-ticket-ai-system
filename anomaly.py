import pandas as pd


def detect_anomalies(data: pd.DataFrame, unresolved_hours: int = 24) -> dict:
    """
    Detect support-ticket anomalies.

    Anomaly types:
    1. Abnormally long resolution times:
       resolution_time_hrs > mean + 2 * standard deviation (Resolved tickets only)
    2. Unresolved (Open/Escalated) high-priority tickets older than `unresolved_hours`.
    """
    anomalies = []

    # 1. Long resolution-time anomalies
    resolved = data[data["resolution_time_hrs"].notna()].copy()

    if not resolved.empty:
        mean_resolution = resolved["resolution_time_hrs"].mean()
        std_resolution = resolved["resolution_time_hrs"].std()
        long_resolution_threshold = mean_resolution + (2 * std_resolution)

        long_resolution = resolved[resolved["resolution_time_hrs"] > long_resolution_threshold]

        for _, row in long_resolution.iterrows():
            anomalies.append({
                "ticket_id": row["ticket_id"],
                "anomaly_type": "abnormally_long_resolution",
                "priority": row["priority"],
                "status": row["status"],
                "resolution_time_hrs": float(row["resolution_time_hrs"]),
                "description": (
                    f"Resolution time of {row['resolution_time_hrs']:.2f} hours "
                    f"exceeds the anomaly threshold of {long_resolution_threshold:.2f} hours."
                )
            })

    # 2. Unresolved high-priority tickets older than N hours
    unresolved = data[data["status"].isin(["Open", "Escalated"])].copy()

    if not unresolved.empty:
        # Use the latest timestamp in the dataset as "now" for reproducibility
        reference_time = data["created_at"].max()
        unresolved["age_hours"] = (reference_time - unresolved["created_at"]).dt.total_seconds() / 3600

        old_high_priority = unresolved[
            unresolved["priority"].isin(["High", "Critical"])
            & (unresolved["age_hours"] > unresolved_hours)
        ]

        for _, row in old_high_priority.iterrows():
            anomalies.append({
                "ticket_id": row["ticket_id"],
                "anomaly_type": "old_unresolved_high_priority",
                "priority": row["priority"],
                "status": row["status"],
                "age_hours": round(float(row["age_hours"]), 2),
                "description": (
                    f"{row['priority']} priority ticket is {row['status']} "
                    f"and is {row['age_hours']:.2f} hours old."
                )
            })

    return {
        "total_anomalies": len(anomalies),
        "anomalies": anomalies
    }
