import argparse, pandas as pd, json, sys
from datetime import datetime
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, help="Path to intake.xlsx")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Intake")
    out = []
    for _, row in df.iterrows():
        out.append({
            "user_id": row.get("user_id","unknown"),
            "timestamp": str(row.get("timestamp") or datetime.utcnow().isoformat()),
            "goal": row.get("goal_sentence","").strip(),
            "target_area": row.get("target_area","").strip(),
            "constraints": row.get("constraints","").strip(),
            "approvals": row.get("approvals","").strip(),
            "model_pref": row.get("model_preference","").strip(),
            "role_overrides": row.get("role_overrides","").strip(),
        })
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
