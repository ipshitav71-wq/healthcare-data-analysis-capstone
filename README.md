# Healthcare Data Analysis Capstone

An integrated project that unites five components of healthcare data analysis in one case study: reducing 30-day hospital readmissions in a four-hospital network.

**Author:** Ipshita Verma

> **Important:** The case study is hypothetical. All patient-level data are **simulated**. No real patients, hospitals or organisations are represented.

## Components

| # | Component | Document |
|---|---|---|
| 1-2 | Data collection strategy and data quality | `docs/Healthcare_Data_Collection_Quality_Strategy.docx` |
| 3 | Data visualization plan | `docs/Healthcare_Data_Visualization_Plan.docx` |
| 4 | Statistical analysis and predictive modeling framework | `docs/Healthcare_Predictive_Modeling_Framework.docx` |
| 5 | Analytical report (simulated case) | `docs/Healthcare_Analytical_Report.docx` |
| Final | **Integrated project report with self-assessment** | `docs/Integrated_Healthcare_Project_Report.docx` (and `.pdf`) |

## Related repositories

- Visualization plan: https://github.com/ipshitav71-wq/healthcare-data-visualization-plan
- Analytical report, simulation and model code: https://github.com/ipshitav71-wq/healthcare-analytics-report-simulated

## Folder structure

| Path | Contents |
|---|---|
| `docs/` | All project documents |
| `images/` | Integrated framework diagram, data-quality figure, dashboard mock-up, and analysis figures |
| `data/` | Simulated cohort (raw and cleaned), `results.json` (numbers used in the report), `qc.json` (quality-check results) |
| `scripts/` | `quality_checks.py` and `integrated_figures.py` |

## How to reproduce the integrated figures and checks

```bash
pip install -r requirements.txt
python scripts/quality_checks.py
python scripts/integrated_figures.py
```

The full simulation and analysis script is in the analytical-report repository above.

## Preview

![Integrated framework](images/fa_lifecycle.png)
![Dashboard mock-up](images/fc_dashboard.png)
