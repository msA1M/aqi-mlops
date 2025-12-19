import pandas as pd

DRIFT_REPORT_PATH = "monitoring/drift_report.csv"

# Thresholds (industry standard)
MODERATE_DRIFT = 0.1
SEVERE_DRIFT = 0.2

def analyze_drift():
    df = pd.read_csv(DRIFT_REPORT_PATH)

    retrain_tasks = []

    for _, row in df.iterrows():
        if row["psi"] >= SEVERE_DRIFT:
            retrain_tasks.append({
                "domain": row["domain"],
                "city": row["city"],
                "feature": row["feature"],
                "severity": "severe"
            })

    return retrain_tasks


def main():
    retrain_tasks = analyze_drift()

    if not retrain_tasks:
        print("✅ No severe drift detected. No retraining required.")
        return

    print("🔥 Retraining triggered due to severe drift:\n")

    for task in retrain_tasks:
        print(
            f"- Domain: {task['domain']} | "
            f"City: {task['city']} | "
            f"Feature: {task['feature']}"
        )

    # Write retraining signal for orchestration
    pd.DataFrame(retrain_tasks).to_csv(
        "monitoring/retrain_signal.csv",
        index=False
    )

    print("\n📌 Retrain signal saved to monitoring/retrain_signal.csv")


if __name__ == "__main__":
    main()
