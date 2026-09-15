# Students Insights

Monorepo of 4 end-to-end student/data projects: grade classification, dropout prediction, attendance regression with weather, and course-review sentiment with DistilBERT.

GitHub currently only shows the old `multiclass/` draft (1M synthetic rows, 70% accuracy, stale 85-95% claims). This local version is the finished work — push this instead.

## Modules

| # | Folder | Problem | Data | Method | Honest result |
|---|--------|---------|------|--------|---------------|
| 1 | `grade_multiclass/` | Multiclass grade A/B/C/D/F from study factors | `Student_Performance.csv`, 10k rows, 5 features (Hours Studied, Previous Scores, Sleep Hours, Practice Papers, Extracurricular) → binned Performance Index | StandardScaler + OneHot, stratified 80/20, 5-fold CV, LR vs RF vs GB, GridSearch on RF/GB (f1_macro), `StudentGradePredictor` class with validation + probabilities + recommendations, `model_artifacts/` | No trusted numbers checked in — `student_performance_classification.py` (1100 lines) is the source of truth, notebook `.ipynb` in same folder is stale (loads lowercase `student_performance.csv`, 1M rows). Run the `.py` to reproduce; do not quote accuracy until then |
| 2 | `dropout_binaryclass/` | Dropout (0) vs Graduate (1), Enrolled filtered out | `data.csv` (UCI, `;` separated), 32 features after dropping 4 | Pipeline StandardScaler + LR balanced, 5-fold stratified CV, coef importance | `model_config.json`: ROC-AUC 0.9426 ±0.0022, acc 0.8904 ±0.0123. Realistic 10-feature version (`realistic/`): ROC 0.9336, acc 0.8857 with XGBoost comparison in `realistic/comparison_xgboost/` |
| 3 | `lr_attendance/` | Predict daily `attendance_rate` | NYC 2018-2019 daily attendance (7.8M raw) + `nyc_weather_2018_2019.csv` → `attendance_features_complete.csv`, school-level avg/std merge | LinearRegression, StandardScaler, single 80/20 split, V1-V4 feature-set ablation (baseline → drop non-sig → +interactions snow_cold, friday_late_year, pre_holiday_friday) | `models/metadata.json` (V3, 13 feats, 221720 train / 55431 test / 1583 schools): R2 0.6094, RMSE 5.94, MAE 3.15 |
| 4 | `course_feedback_nlp/` | 5-star sentiment → also 3-class (neg/neu/pos) teacher | `Coursera_reviews.csv` (~1.5M rows per git), 78.8% 5-star | DistilBERT fine-tune local `./distilbert-base-uncased`, max_len 96, batch 128, 5 epochs, AdamW 2e-5, linear warmup 10%, AMP, class-weighted CE, pretok + TensorDataset, checkpoints every epoch | `teacher_sentiment_model/results.json` (3-class): test acc 96.16%, macro-F1 83.27, weighted-F1 96.65, missed struggling 5.64%. 5-class `sentiment_model/results.pt` exists but needs re-eval with matching torch version — do not quote until re-run |

## Repo layout (finished)

```
grade_multiclass/
  Student_Performance.csv
  student_performance_classification.py  # finished — use this, not .ipynb
  model_artifacts/  # model.pkl 27M + preprocessor.pkl
  web/              # index.html/script.js/style.css demo
  *.png             # 01-11 EDA + CV plots
dropout_binaryclass/
  data.csv  train.py  train.ipynb
  model_config.json  student_dropout_model.pkl
  realistic/  # 10 practical feats + train.py
  realistic/comparison_xgboost/  # LR vs XGB
lr_attendance/
  train.py  feature_engineering.py  prepare_for_modeling.py
  models/metadata.json  models/model.joblib
course_feedback_nlp/
  train.py  train_3_classes.py  evaluate.py  test.py
  teacher_sentiment_model/results.json
```

## Reproduce (each independent)

```bash
# 1. grades (10k — fast)
cd grade_multiclass
pip install pandas scikit-learn matplotlib seaborn joblib
python student_performance_classification.py

# 2. dropout
cd ../dropout_binaryclass
pip install pandas scikit-learn matplotlib seaborn joblib xgboost
python train.py
python realistic/train.py

# 3. attendance (needs 60M+ CSVs, ~277k clean rows)
cd ../lr_attendance
pip install pandas scikit-learn joblib
python train.py

# 4. NLP (needs GPU, 24GB VRAM class, local distilbert-base-uncased/)
cd ../course_feedback_nlp
pip install torch transformers scikit-learn pandas matplotlib seaborn tqdm
python train.py
python train_3_classes.py
python evaluate.py
```

Web demos: `grade_multiclass/web/` and old `multiclass/web/` are static — `python -m http.server 8000` inside folder.

## Author

Nizar Sahl — https://github.com/nmetal05/students-insights — HF mirror `sahlnizar/students-insights`
