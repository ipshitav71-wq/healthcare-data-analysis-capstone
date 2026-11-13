import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
R = json.load(open("data/results.json")); df = pd.read_csv("data/simulated_clean_cohort.csv")
BLUE="#0072B2"; ORANGE="#E69F00"; VERM="#D55E00"; GREEN="#009E73"; SKY="#56B4E9"; PURPLE="#CC79A7"; INK="#222222"; GREY="#6B6B6B"; LINE="#C9CED3"; LIGHT="#F3F5F7"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})

# ---------- Fig A: integrated lifecycle ----------
fig, ax = plt.subplots(figsize=(10.5, 5.6)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(5.25, 5.35, "Integrated framework: how the five components connect", ha="center", fontsize=11.5, fontweight="bold")
boxes = [("1 Data strategy","Sources, selection\ncriteria, access, ethics",SKY,0.2,3.3),
         ("2 Data quality","Five-dimension checks,\ncleaning, risk register",SKY,2.3,3.3),
         ("3 Analysis and\nmodelling","Trends, drivers,\nprediction, validation",GREEN,4.4,3.3),
         ("4 Visualization","Dashboards, briefing,\naccessible charts",ORANGE,6.5,3.3),
         ("5 Reporting and\nrecommendations","Evidence-linked actions,\nlimitations",VERM,8.6,3.3)]
for t, d, c, x, y in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 1.75, 1.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=2))
    ax.add_patch(Rectangle((x, y+0.92), 1.75, 0.58, fc=c, ec="none"))
    ax.text(x+0.875, y+1.21, t, ha="center", va="center", fontsize=8.5, fontweight="bold", color="white" if c in (GREEN, VERM) else INK)
    ax.text(x+0.875, y+0.45, d, ha="center", va="center", fontsize=7.6)
for x in (1.97, 4.07, 6.17, 8.27):
    ax.add_patch(FancyArrowPatch((x, 4.05), (x+0.31, 4.05), arrowstyle="-|>", mutation_scale=11, color=GREY))
# decision + monitoring
ax.add_patch(FancyBboxPatch((2.35, 1.45), 5.8, 0.85, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#F3F5F7", ec=INK, lw=1.4))
ax.text(5.25, 1.87, "Decision and action\n(follow-up capacity, discharge process, equity programme)", ha="center", va="center", fontsize=8.6, fontweight="bold")
for x in (1.07, 3.17, 5.27, 7.37, 9.47):
    ax.add_patch(FancyArrowPatch((x, 3.28), (x if x > 2 else 3.0, 2.32) if False else (min(max(x, 2.7), 7.8), 2.32), arrowstyle="-|>", mutation_scale=9, color=LINE, lw=1.2))
ax.add_patch(FancyArrowPatch((5.25, 1.43), (5.25, 0.9), arrowstyle="-|>", mutation_scale=11, color=GREY))
ax.add_patch(FancyBboxPatch((1.2, 0.2), 8.1, 0.68, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=GREY, lw=1.2, ls="--"))
ax.text(5.25, 0.54, "Monitoring and learning: data-quality alerts, model drift and equity checks, user feedback", ha="center", va="center", fontsize=8.6)
ax.add_patch(FancyArrowPatch((1.2, 0.54), (0.35, 0.54), arrowstyle="-", color=GREY, lw=1)); ax.add_patch(FancyArrowPatch((0.35, 0.54), (0.35, 3.25), arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1, ls="--"))
ax.text(0.28, 2.0, "feedback loop", rotation=90, fontsize=7.6, color=GREY, ha="right", va="center")
ax.text(5.25, 2.55, "Ethics, privacy and accessibility apply to every component", ha="center", fontsize=8.4, color=VERM, style="italic")
fig.savefig("images/fa_lifecycle.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------- Fig B: data quality at a glance ----------
fl = R["flow"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]
steps = [("Raw extract", fl["raw"]), ("Duplicates removed", -fl["dups"]), ("In-hospital deaths", -fl["died"]), ("Hospice discharges", -fl["hospice"]), ("Transfers out", -fl["transfer"]), ("Missing length of stay", -fl["los_missing"]), ("Analytic cohort", fl["final"])]
run = 0; labels = []
for i, (lab, v) in enumerate(steps):
    if i == 0: ax.bar(i, v, color=SKY); run = v; ax.text(i, v+250, f"{v:,}", ha="center", fontsize=8, fontweight="bold")
    elif i == len(steps)-1: ax.bar(i, v, color=BLUE); ax.text(i, v+250, f"{v:,}", ha="center", fontsize=8, fontweight="bold")
    else: ax.bar(i, -v, bottom=run+v, color=VERM); ax.text(i, run+250, f"\u2212{-v:,}", ha="center", fontsize=8, color=VERM, fontweight="bold"); run += v
    labels.append(lab)
ax.set_xticks(range(len(steps))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8); ax.set_ylim(22000, 26500); ax.set_ylabel("Records (axis starts at 22,000)")
ax.set_title("From raw extract to analytic cohort", loc="left", fontsize=9.5, fontweight="bold"); ax.grid(axis="y", color="#E5E8EB", lw=0.8)
ax = axs[1]; mp = R["missing_pct"]; names = ["Income group", "Comorbidity count", "Medication reconciliation"]; vals = [mp["income"], mp["comorb"], mp["medrec"]]
ax.barh(range(3)[::-1], vals, color=ORANGE, height=0.55)
for i, v in enumerate(vals): ax.text(v+0.15, 2-i, f"{v}%", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(range(3)[::-1]); ax.set_yticklabels(names); ax.set_xlim(0, 9.5); ax.set_xlabel("Missing values (% of cohort)")
ax.set_title("Missing data after cleaning", loc="left", fontsize=9.5, fontweight="bold"); ax.grid(axis="x", color="#E5E8EB", lw=0.8)
fig.tight_layout(); fig.savefig("images/fb_quality.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------- Fig C: integrated dashboard with real case numbers ----------
feat_num = ["age", "comorb", "prior_adm", "los", "male", "weekend", "comorb_missing"]; feat_cat = ["dest", "ward", "hospital", "medrec_cat"]
Cbest = float(R["best_lr"].split("=")[1])
pre = ColumnTransformer([("num", StandardScaler(), feat_num), ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat)])
tr = df.month <= 17; te = df.month >= 21
m = Pipeline([("pre", pre), ("clf", LogisticRegression(C=Cbest, max_iter=2000))]).fit(df.loc[tr, feat_num+feat_cat], df.loc[tr, "readmit30"])
dt = df[te].copy(); dt["p"] = m.predict_proba(dt[feat_num+feat_cat])[:, 1]; top = dt.sort_values("p", ascending=False).head(5)
fig = plt.figure(figsize=(11.2, 7.4), facecolor=LIGHT)
fig.text(0.02, 0.955, "Readmission Monitoring Dashboard (case-study data)", fontsize=15, fontweight="bold")
fig.text(0.02, 0.925, "Simulated data  |  Period: Sep 2024 - Aug 2026  |  Refreshed: 24 Oct 2026  |  Filters: Hospital, Ward, Period, Patient group", fontsize=8.5, color=GREY)
def tile(x, label, val, note, st):
    ax = fig.add_axes([x, 0.78, 0.232, 0.115]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec=LINE))
    c = {"good": BLUE, "watch": ORANGE, "alert": VERM}[st]; mk = {"good": "\u25cf", "watch": "\u25b2", "alert": "\u25a0"}[st]
    ax.add_patch(Rectangle((0, 0), 0.035, 1, fc=c, ec="none")); ax.text(0.09, 0.78, label, fontsize=8.5, color=GREY); ax.text(0.09, 0.42, val, fontsize=19, fontweight="bold"); ax.text(0.09, 0.12, f"{mk} {note}", fontsize=8, color=c, fontweight="bold")
tile(0.02, "Network readmission, last 6 months", f"{R['rate_last6']}%", "steady vs earlier", "good")
tile(0.265, "Hospital D readmission, last 6 months", f"{R['D_rates'][1]}%", f"was {R['D_rates'][0]}% earlier", "alert")
tile(0.51, "Follow-up within 7 days, Hospital D", f"{R['fu_D_last6']}%", f"was {R['fu_D_early']}% (target 55%)", "alert")
tile(0.755, "Medication reconciliation, Hospital D", f"{R['medrec_D']}%", f"others {R['medrec_ABC']}%", "watch")
# D chart
dD = df[df.hospital == "D"]; t = dD.groupby("month").readmit30.agg(["sum", "count"]); t["rate"] = t["sum"]/t["count"]
pb = t.loc[0:11, "sum"].sum()/t.loc[0:11, "count"].sum(); ucl = pb + 3*np.sqrt(pb*(1-pb)/t["count"])
ax = fig.add_axes([0.05, 0.43, 0.41, 0.28], facecolor="white")
ax.plot(t.index, 100*t.rate, color=BLUE, lw=2, marker="o", ms=3.5); ax.axhline(100*pb, color=GREY, lw=1); ax.step(t.index, 100*ucl, where="mid", color=VERM, ls="--", lw=1)
ax.scatter(t.index[t.rate > ucl], 100*t.rate[t.rate > ucl], color=VERM, marker="s", s=45, zorder=5)
ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["Sep 24", "Mar 25", "Sep 25", "Mar 26", "Aug 26"], fontsize=8); ax.set_ylabel("% readmitted"); ax.set_ylim(5, 26); ax.grid(axis="y", color="#E5E8EB", lw=0.8)
ax.set_title("Hospital D: rate climbing above baseline", loc="left", fontsize=9.5, fontweight="bold")
# follow-up lines
ax = fig.add_axes([0.54, 0.43, 0.42, 0.28], facecolor="white")
g = df.assign(grp=np.where(df.hospital == "D", "D", "ABC")).groupby(["grp", "month"]).followup7.mean().unstack(0)
ax.plot(g.index, 100*g["ABC"], color=BLUE, lw=2, label="Hospitals A-C"); ax.plot(g.index, 100*g["D"], color=VERM, lw=2, ls="--", label="Hospital D"); ax.axhline(55, color=GREY, lw=1, ls=":")
ax.text(5, 56.5, "Target 55%", fontsize=7.5, color=GREY); ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["Sep 24", "Mar 25", "Sep 25", "Mar 26", "Aug 26"], fontsize=8); ax.set_ylim(15, 75); ax.set_yticks([20, 30, 40, 50, 60, 70]); ax.set_ylabel("% with follow-up")
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(axis="y", color="#E5E8EB", lw=0.8); ax.set_title("7-day follow-up fell only in Hospital D", loc="left", fontsize=9.5, fontweight="bold")
# bars
ax = fig.add_axes([0.05, 0.07, 0.28, 0.28], facecolor="white")
for i, (v, c) in enumerate(zip([R["fu_yes"], R["fu_no"]], [BLUE, VERM])):
    ax.bar(i, v[0], color=c, width=0.55); ax.errorbar(i, v[0], yerr=[[v[0]-v[1]], [v[2]-v[0]]], color=INK, capsize=4); ax.text(i, v[2]+0.5, f"{v[0]:.1f}%", ha="center", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Follow-up\nwithin 7 days", "No follow-up\nwithin 7 days"], fontsize=8); ax.set_ylim(0, 24); ax.set_yticks([0, 5, 10, 15, 20]); ax.set_ylabel("% readmitted")
ax.set_title("Readmission is twice as high\nwithout follow-up", loc="left", fontsize=9.5, fontweight="bold")
# priority list
ax = fig.add_axes([0.38, 0.07, 0.58, 0.28]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc="white", ec=LINE))
ax.text(0.03, 0.92, "Priority call list: highest predicted risk, Jun-Aug 2026 discharges (top 5 of model output)", fontsize=9, fontweight="bold")
cols = [("Encounter", 0.03), ("Age", 0.20), ("Ward", 0.30), ("Hospital", 0.50), ("Prior adm.", 0.63), ("Predicted risk", 0.80)]
for h, x in cols: ax.text(x, 0.78, h, fontsize=7.8, color=GREY)
for i, (_, r) in enumerate(top.iterrows()):
    y = 0.64 - i*0.125
    vals = [f"E{int(r.encounter_id)}", f"{int(r.age)}", r.ward, r.hospital, f"{int(r.prior_adm)}", f"\u25a0 {100*r.p:.0f}%"]
    for (h, x), v in zip(cols, vals): ax.text(x, y, v, fontsize=8.3, color=VERM if h == "Predicted risk" else INK, fontweight="bold" if h == "Predicted risk" else "normal")
    ax.plot([0.03, 0.97], [y-0.055, y-0.055], color="#E5E8EB", lw=0.8)
ax.text(0.03, 0.02, "Role-based access; clinicians decide who is called. Model AUROC %.2f, a prioritisation aid only." % R["auc_lr"][0], fontsize=7.8, color=GREY, style="italic")
for n, (x, y) in enumerate([(0.015, 0.895), (0.04, 0.715), (0.525, 0.715), (0.04, 0.355), (0.365, 0.355)], 1):
    fig.text(x, y, str(n), fontsize=8, fontweight="bold", color="white", ha="center", va="center", bbox=dict(boxstyle="circle,pad=0.25", fc=INK, ec="none"), zorder=30)
fig.savefig("images/fc_dashboard.png", dpi=200, bbox_inches="tight", facecolor=LIGHT); plt.close(fig)
print("ok", top[["encounter_id","age","ward","hospital","prior_adm","p"]].round(3).to_dict("records"))
