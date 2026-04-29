# CARE to Compare: Wind Farm Anomaly Detection

This project applies machine learning to wind turbine SCADA data from the CARE to Compare benchmark. The goal is to detect anomaly periods in Wind Farm C and evaluate whether the model can support predictive maintenance by identifying abnormal turbine behaviour before or during known event windows.

The work was originally developed as an academic machine learning project. It contains feature engineering, supervised anomaly detection, unsupervised clustering experiments, CARE score evaluation, and SHAP-based interpretability for model traceability.

## Project Overview

Wind turbine SCADA data is high-dimensional, noisy, and partially anonymised. Each dataset contains time-series measurements sampled at 10-minute intervals, including turbine metadata, train/prediction split information, status IDs, wind speed, power, reactive power, and anonymised sensor signals.

The project explores two approaches:

1. **Supervised learning**
   - Uses a Random Forest classifier.
   - Training labels are created from refined anomaly candidates in the training frame.
   - Prediction windows are evaluated against event ranges from `event_info.csv`.
   - Uses custom probability thresholds to control false positives.

2. **Unsupervised learning**
   - Uses PCA, UMAP, and HDBSCAN.
   - Detects structural outliers based on cluster confidence, distance from cluster centroids, and small/noisy clusters.
   - Useful for comparison, but more sensitive to hyperparameters and less stable across turbines.

The supervised approach was preferred because it was more stable, easier to interpret, and more directly connected to wind and power behaviour.

## Dataset

The project uses Wind Farm C data from the CARE to Compare benchmark:

- Individual turbine/event CSV files are stored in `datasets/`.
- `event_info.csv` contains anomaly event windows and event descriptions.
- `feature_description.csv` contains metadata for sensor groups, units, and descriptions.

Important dataset columns include:

- `time_stamp`: timestamp of the 10-minute SCADA sample
- `asset_id`: turbine identifier
- `id`: sample index
- `train_test`: whether the row belongs to the training or prediction frame
- `status_type_id`: turbine status code
- `event_start_id`, `event_end_id`: event window boundaries from `event_info.csv`

Many sensor names and labels are anonymised, so model interpretation is treated as a traceability check rather than a confirmed physical root-cause diagnosis.

## Feature Engineering

The original feature space contains average, minimum, maximum, and standard deviation values for many sensors. The preprocessing pipeline reduces this space and keeps the most useful statistical signals.

Main preprocessing steps:

- Drop minimum and maximum columns because the sampling interval is 10 minutes and isolated min/max readings may add noise.
- Keep average and standard deviation features.
- Merge correlated or duplicate sensor groups.
- Prioritise sensor groups relevant to known anomaly types, including:
  - wind speed
  - power output
  - reactive power
  - pitch and rotor brake sensors
  - hydraulic system sensors
  - gearbox and mechanical sensors
  - electrical and converter sensors
  - yaw and axis sensors
- Standardise numeric features using `StandardScaler`.

After preprocessing, the feature space is reduced to approximately 240 numeric features.

## Supervised Approach

The supervised pipeline is implemented mainly in `datasets/WP1.ipynb`.

The main idea is to use normal-status training data and identify deviations from expected turbine behaviour. Wind speed and power output are especially important because they are physically meaningful and less anonymised than many other sensors.

### Label Creation

The training labels are not direct fault labels. They are generated from refined anomaly candidates:

1. Select training samples with `status_type_id == 0`.
2. Compare wind speed and power behaviour against expected turbine operation.
3. Use DBSCAN to separate normal points from deviating points.
4. Use KDE-based inspection to remove overlapping or saturated power regions.
5. Mark cleaned deviations as anomaly class `1`; all other points are normal class `0`.

For testing, known event ranges from `event_info.csv` are used to construct binary ground-truth windows.

## Model

The supervised model is a Random Forest classifier:

```python
RandomForestClassifier(
    n_estimators=180,
    max_depth=12,
    random_state=42,
    min_samples_split=22,
    min_samples_leaf=11,
    max_features="sqrt"
)
```

Random Forest was selected because it is robust to noise, handles high-dimensional tabular data well, supports feature importance analysis, and works effectively with non-linear sensor relationships.

Predictions are generated using `predict_proba()` instead of hard labels. This allows threshold tuning, for example:

```python
threshold = 0.6
y_score = rf.predict_proba(X_test)[:, 1]
y_pred = (y_score > threshold).astype(int)
```

Using probability thresholds helps reduce false positives while still allowing early anomaly detection.

## CARE Score Evaluation

Model performance is evaluated using the CARE framework, which combines:

- **Coverage**: how well anomaly points are detected
- **Accuracy**: how well normal operation is preserved
- **Reliability**: event-level detection quality
- **Earliness**: how early the anomaly is detected inside the event window

The custom CARE implementation is available in `datasets/model.py`.

Example supervised results from the submitted report:

| Dataset | Lead Time | Confidence | Coverage | Accuracy | Reliability | Earliness | CARE Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| 44 benchmark | 0d 4h 30m | 0.610 | 0.222 | 0.934 | 0.077 | 0.045 | 0.264 |
| 49 | 1d 18h 50m | 0.660 | 0.146 | 0.938 | 0.088 | 0.081 | 0.266 |
| 15 | 0d 19h 0m | 0.930 | 0.483 | 0.795 | 0.200 | 0.140 | 0.352 |
| 67 | 0d 22h 10m | 0.960 | 0.577 | 0.689 | 0.135 | 0.209 | 0.364 |
| 9 | 1d 2h 40m | 0.450 | 0.277 | 0.995 | 0.385 | 0.056 | 0.354 |
| 76 | 1d 21h 50m | 0.830 | 0.044 | 0.823 | 0.088 | 0.048 | 0.210 |

The model produced useful early-warning behaviour on several datasets, but performance varies by turbine/event because the event windows, operating modes, and anomaly development patterns differ.

## SHAP Interpretability

SHAP was added to improve model traceability and explainability.

The goal of SHAP in this project is not to claim exact physical root cause, because many labels and sensor names are anonymised. Instead, SHAP is used to check whether the Random Forest relies on operationally reasonable features and to explain why individual predictions are flagged as anomalous.

The SHAP analysis includes:

- A global feature importance bar plot using mean absolute SHAP values.
- A beeswarm plot showing whether high or low feature values push the model toward anomaly.
- A local waterfall plot explaining one high-risk prediction.

The top SHAP features included signals such as:

- `wind_speed_235_merged_avg`
- `sensor_100_merged_avg`
- `sensor_26_avg`
- `sensor_76_std`
- `sensor_76_avg`
- `reactive_power_120_avg`
- `power_2_merged_avg`

This is a useful result because wind speed, power, reactive power, and sensor variability are signals that humans would also expect to matter in wind turbine anomaly detection.

In practical terms, SHAP changes the output from:

```text
prediction = anomaly
```

to a more traceable explanation:

```text
prediction = anomaly because wind speed, power, reactive power,
and selected sensor features pushed the anomaly probability upward.
```

This makes the model easier to inspect, explain, and audit.

## Unsupervised Approach

The unsupervised pipeline is explored in `datasets/WP_Unsuper.ipynb`.

Main steps:

1. Downsample 10-minute data to hourly resolution to reduce noise.
2. Apply PCA to reduce redundant variance.
3. Apply UMAP for non-linear embedding.
4. Use HDBSCAN for density-based clustering.
5. Mark anomalies based on:
   - HDBSCAN noise labels
   - weak cluster confidence
   - distance from cluster centroid
   - very small clusters

The unsupervised method performed well on some benchmark cases but was less stable across turbines. It was sensitive to hyperparameter settings and sometimes produced high false-positive rates.

Example unsupervised CARE results:

| Dataset | Coverage | Accuracy | Reliability | Earliness | CARE Score |
|---|---:|---:|---:|---:|---:|
| 44 benchmark | 0.9709 | 0.6467 | 1.0000 | 0.2906 | 0.6397 |
| 79 | 0.6991 | 0.6620 | 0.0000 | 0.0000 | 0.2722 |
| 76 | 0.5013 | 0.5000 | 0.0000 | 0.0000 | 0.2003 |
| 90 | 0.8765 | 0.5053 | 1.0000 | 0.1093 | 0.5201 |

## Repository Structure

```text
.
|-- README.md
|-- event_info.csv
|-- feature_description.csv
|-- NOtes.docx
`-- datasets/
    |-- WP1.ipynb              # Main supervised pipeline and SHAP analysis
    |-- WP_Unsuper.ipynb       # Unsupervised PCA/UMAP/HDBSCAN experiments
    |-- main.ipynb             # Additional experiments
    |-- utils.py               # Preprocessing and feature utilities
    |-- model.py               # CARE score and anomaly detection helpers
    |-- my_random_forest.pkl   # Saved Random Forest model
    `-- *.csv                  # Wind Farm C turbine/event data files
```

## How To Run

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

For the unsupervised notebook, also install:

```bash
pip install umap-learn hdbscan optuna plotly
```

Open the supervised notebook:

```bash
jupyter notebook datasets/WP1.ipynb
```

If imports such as `from utils import ...` fail, start Jupyter from inside the `datasets/` directory or add the `datasets` folder to the Python path.

## Key Takeaways

- The supervised Random Forest approach was more stable and interpretable than the unsupervised clustering approach.
- Wind speed and power-related signals were central to anomaly detection.
- CARE score evaluation made it possible to assess not only classification quality, but also reliability and early detection.
- SHAP improved traceability by showing which features influenced both global model behaviour and individual anomaly predictions.
- Because the labels and sensors are partially anonymised, the model should be interpreted as an anomaly detection and traceability tool, not as a definitive root-cause diagnosis system.

## References

- Guck, C., Roelofs, C., and Faulstich, S. "CARE to Compare: A real-world dataset for anomaly detection in wind turbine data." arXiv preprint arXiv:2404.10320, 2024.
- Nguyen, W. "Wind Power Curve Modeling." Kaggle, 2022.
- Shokrzadeh, S., Jafari Jozani, M., and Bibeau, E. "Wind turbine power curve modeling using advanced parametric and nonparametric methods." IEEE Transactions on Sustainable Energy, 2014.
- Brownlee, J. "Threshold Moving for Imbalanced Classification." Machine Learning Mastery, 2020.
- Lun, A. T. L., McCarthy, D. J., and Hicks, S. C. "Dimensionality Reduction." Orchestrating Single-Cell Analysis with Bioconductor, 2022.
