import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# PATHS
# -----------------------
DRIFT_REPORT_PATH = "monitoring/drift_report.csv"
OUTPUT_DIR = Path("monitoring/drift_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv(DRIFT_REPORT_PATH)

# -----------------------
# PLOT
# -----------------------
for domain in df["domain"].unique():
    domain_df = df[df["domain"] == domain]

    for city in domain_df["city"].unique():
        city_df = domain_df[domain_df["city"] == city]

        plt.figure(figsize=(8, 4))
        plt.bar(city_df["feature"], city_df["psi"])
        plt.axhline(0.1, color="orange", linestyle="--", label="Moderate Drift")
        plt.axhline(0.2, color="red", linestyle="--", label="Severe Drift")

        plt.title(f"{domain} Drift — {city}")
        plt.ylabel("PSI")
        plt.xlabel("Feature")
        plt.legend()
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{domain}_{city}_drift.png"
        plt.savefig(output_path)
        plt.close()

        print(f"📊 Saved {output_path}")
