"""
Convert Kaggle FitBit heartrate_seconds_merged.csv to simple format (timestamp,hr).

Usage:
    python scripts/prepare_fitbit_kaggle.py --input /path/to/heartrate_seconds_merged.csv --output data/fitbit_mapped.csv
"""
import argparse, pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()

df = pd.read_csv(args.input)
if not {'Time','Value'}.issubset(df.columns):
    raise SystemExit("Expected columns 'Time' and 'Value' in Kaggle file.")
out = pd.DataFrame({'timestamp': pd.to_datetime(df['Time'], utc=True), 'hr': df['Value']}).sort_values('timestamp')
out.to_csv(args.output, index=False)
print(f"Wrote {args.output} with {len(out)} rows.")
