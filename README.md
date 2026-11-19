# Predicting Term Deposit Subscriptions in Bank Telemarketing Campaigns

**Course:** Tools and Techniques of Data Science — End Semester Project  
**Dataset:** [UCI Bank Marketing Dataset (ID 222, `bank-additional-full.csv`)](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**Problem:** Binary Classification (`target`: `y`, `id_column`: `row_id`, `primary_metric`: `f1_macro`)

## Repository Layout

- `project.json` — Registration manifest with group ownership, target (`y`), ID column (`row_id`), primary metric (`f1_macro`), and dataset row/feature counts (`41,188` rows, `20` raw features).
- `run_all.py` — End-to-end reproducible entry point (`SEED = 808`). Splits row IDs (`75%` train / `25%` test), fits the `pyspark.ml.Pipeline` (`Imputer`, `StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`) strictly on training rows, writes `results/spark_plan.txt`, evaluates `spark_logistic` (Spark MLlib), `sklearn_boosting` (`HistGradientBoostingClassifier`), and `majority_class` (baseline) with 1,000-sample 95% bootstrap confidence intervals, and outputs `results/metrics.csv`.
- `data/bank_marketing.csv` — Dataset file (`41,188` rows, `20` features + `row_id` + `y`).
- `src/download_bank_data.py` — Script to fetch the raw `bank-additional-full.csv` directly from the UCI repository on an internet-connected machine.
- `src/eda_plots.py` — Reproduces the 4 exploratory data analysis and model comparison figures used in `report.pdf` and `slides.pdf`.
- `results/` — Generated split IDs (`train_ids.csv`, `test_ids.csv`), Spark Catalyst execution plan (`spark_plan.txt`), and model evaluation table (`metrics.csv`).
- `report.pdf` — Technical report covering problem formulation, data leakage removal (`duration`), Spark preprocessing, EDA findings, model comparison, and reflection.
- `slides.pdf` & `slides.pptx` — Presentation slides for the viva defense.

## Running & Checking the Project

```bash
python run_all.py
python check_project.py . --rerun
```
