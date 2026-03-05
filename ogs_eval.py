import pandas as pd
import json
import re
import os
import numpy as np

# --- CONFIG ---
RESULTS_JSON = "/predictions.json"
GT_CSV = "/annotations.json"

name = 'cerebral_palsy'

# Mapping the JSON keys from Cosmos to your Excel Column Names
# Right side uses the base name; Left side uses the '.1' suffix added by pandas

# def extract_scores(text):
#     match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', text, re.DOTALL)
#     return json.loads(match.group(1)) if match else None


def extract_scores(text):
    """Extracts and repairs JSON answer from <answer> tags."""
    try:
        # 1. Isolate the content between tags
        match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', text, re.DOTALL)
        if not match:
            # Fallback: find anything that looks like a JSON block if tags are missing
            match = re.search(r'(\{.*\})', text, re.DOTALL)

        if match:
            json_str = match.group(1).strip()

            # 2. Common VLM Fixes:
            # Fix missing commas between key-value pairs (e.g., "key": 3 "key2": 1)
            json_str = re.sub(r'(\d|")\s+"', r'\1, "', json_str)
            # Fix trailing commas before closing braces
            json_str = re.sub(r',\s*}', '}', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Last ditch effort: simple python evaluation if it's almost-JSON
                # (Be careful with this, but for hackathon results it's a lifesaver)
                import ast
                try:
                    return ast.literal_eval(json_str)
                except:
                    return None
    except Exception as e:
        print(f"Repair failed: {e}")
    return None

# 1. Load Ground Truth
# We skip the first row to get clean column names
gt_df = pd.read_excel(GT_CSV)
#gt_df.rename(columns={gt_df.columns[0]: 'Patient ID'}, inplace=True)

# 2. Load Cosmos Results
with open(RESULTS_JSON, "r") as f:
    cosmos_results = json.load(f)

print(f"Analyzing {len(cosmos_results)} videos...")

comparison = []

for vid_name, raw_text in cosmos_results.items():
    # Extract Patient ID (e.g., 'p31')
    id_match = re.search(r'cropped_([^_]+)', vid_name)
    if id_match:
        p_id = id_match.group(1)  # This will correctly return '100-14'

    pred = extract_scores(raw_text)
    if not pred: continue

    # Get GT row
    if p_id not in gt_df['Patient ID'].values:
        print(f"Patient {p_id} not found in ground truth. Skipping...")
        continue
    gt_row = gt_df[gt_df['Patient ID'] == p_id]
    if gt_row.empty: continue

    # --- CALCULATE TOTALS ---
    # We sum ONLY the categories predicted by the model for a fair comparison
    pred_r_total = sum(v for k, v in pred['right'].items() if isinstance(v, (int, float)))
    pred_l_total = sum(v for k, v in pred['left'].items() if isinstance(v, (int, float)))

    # Sum corresponding 6 columns from Excel for Right
    actual_r_total = gt_row["R_OGS_6Params"].values[0]
    actual_l_total = gt_row["L_OGS_6Params"].values[0]

    # Symmetry Comparison
    actual_sym = gt_row['Symmetry (1)/Asymmetry (0)'].values[0]
    pred_sym = pred.get('symmetry', 0)
    corrected_pred_sym = 1 - pred_sym

    comparison.append({
        "Patient": p_id,
        "Actual_R": actual_r_total, "Pred_R": pred_r_total,
        "Actual_L": actual_l_total, "Pred_L": pred_l_total,
        "Actual_Sym": actual_sym, "Pred_Sym": corrected_pred_sym
    })

# 3. Final Metrics
eval_df = pd.DataFrame(comparison)
eval_df['Error_R'] = abs(eval_df['Actual_R'] - eval_df['Pred_R'])
eval_df['Error_L'] = abs(eval_df['Actual_L'] - eval_df['Pred_L'])

print("-" * 30)
print(f"FINAL MAE REPORT")
print(f"Right OGS MAE: {eval_df['Error_R'].mean():.4f}")
print(f"Left OGS MAE:  {eval_df['Error_L'].mean():.4f}")
print(f"Symmetry Accuracy: {(eval_df['Actual_Sym'] == eval_df['Pred_Sym']).mean() * 100:.2f}%")
print("-" * 30)

eval_df.to_csv(f"{name}_evaluation.csv", index=False)

# Create a summary dictionary
summary_data = {
    "Metric": [
        "Right OGS MAE",
        "Left OGS MAE",
        "Overall Mean Absolute Error",
        "Symmetry Accuracy (%)",
        "Total Samples Validated"
    ],
    "Value": [
        eval_df['Error_R'].mean(),
        eval_df['Error_L'].mean(),
        (eval_df['Error_R'].mean() + eval_df['Error_L'].mean()) / 2,
        (eval_df['Actual_Sym'] == eval_df['Pred_Sym']).mean() * 100,
        len(eval_df)
    ]
}

# Convert to DataFrame and save
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f"{name}_summary_report.csv", index=False)

print(f"Summary report saved to: clinical_summary_report.csv")