# ML Fraud Detection

##Project Description

Detecting Synthetic credit card fruad using Machine learning models to identify suspicius transactions patterns and reduce financial fraud
This project applies RandomForest and XGBoost classifiers to real-world transactions data, with a focus on improving recall for fraud detection.
It includes preprocessing, SMOTE oversampling, feature engineering and hyperparameter tuning.
The goal is to identify fraud cases accurately  while minimizing false positives.




├── LICENSE
├── Makefile             <- Commands like `make data` or `make train`
├── README.md            <- Project overview and instructions (this file)
├── data
│   ├── raw              <- Original, immutable data dump
│   ├── external         <- Third-party or externally sourced data
│   ├── interim          <- Data after cleaning or minimal transformation
│   └── processed        <- Final data used for modeling
│
├── docs                 <- Sphinx documentation (optional)
│
├── models               <- Trained and serialized models, logs, summaries
│
├── notebooks            <- Jupyter notebooks for exploration and experiments
│                         Naming: `1.0-xyz-task-name.ipynb`
│
├── references           <- Data dictionaries, manuals, and explanatory docs
│
├── reports
│   ├── figures          <- Visualizations generated for reports
│   └── ...              <- PDF/HTML reports or presentations
│
├── requirements.txt     <- Python dependencies
├── setup.py             <- Makes `src` a pip-installable module
├── tox.ini              <- For automated testing with `tox`
│
├── src                  <- Source code
│   ├── __init__.py      <- Makes `src` a Python module
│   ├── data
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── features
│   │   └── feature_engineering.py
│   ├── models
│   │   ├── train_models.py
│   │   └── evaluate_models.py
│   └── visualization
│       └── visualization.py



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
