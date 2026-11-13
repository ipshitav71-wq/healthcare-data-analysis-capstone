"""Quality checks on the simulated cohort (run from the repository root)."""
import pandas as pd, json
df = pd.read_csv("data/simulated_clean_cohort.csv"); raw = pd.read_csv("data/simulated_raw_extract.csv")
q = {}
q["dup_ids_clean"] = int(df.encounter_id.duplicated().sum()); q["dup_rows_raw"] = int(raw.encounter_id.duplicated().sum())
q["age_range"] = [int(df.age.min()), int(df.age.max())]; q["age_out"] = int(((df.age < 18) | (df.age > 100)).sum())
q["los_range"] = [int(df.los.min()), int(df.los.max())]; q["los_out"] = int(((df.los < 1) | (df.los > 60)).sum())
q["comorb_range"] = [float(df.comorb.min()), float(df.comorb.max())]; q["prior_range"] = [int(df.prior_adm.min()), int(df.prior_adm.max())]
q["dest_valid"] = bool(set(df.dest.unique()) <= {"home", "home_services", "facility"})
q["ward_valid"] = bool(set(df.ward.unique()) <= {"Medical", "Surgical", "Cardiac", "Respiratory", "Orthopaedic"})
q["binary_ok"] = bool(all(set(df[c].unique()) <= {0, 1} for c in ["readmit30", "followup7", "weekend", "male", "rural"]))
q["month_ok"] = bool(df.month.between(0, 23).all())
g = df.groupby("patient_id").agg(a=("age", "nunique"), s=("male", "nunique"), r=("rural", "nunique"))
q["pat_inconsistent"] = int(((g.a > 1) | (g.s > 1) | (g.r > 1)).sum()); q["n_pat_multi"] = int((df.groupby("patient_id").size() > 1).sum())
q["excl_flags_left"] = int(((df.died == 1) | (df.hospice == 1) | (df.transfer == 1)).sum())
q["raw_miss_income_rural"] = [round(100*float(raw[raw.rural == 0].income.isna().mean()), 1), round(100*float(raw[raw.rural == 1].income.isna().mean()), 1)]
q["raw_n"] = len(raw)
json.dump(q, open("data/qc.json", "w"), indent=1); print(q)
