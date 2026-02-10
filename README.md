# Customer Churn Prediction – End-to-End ML Pipeline

An industry-style machine learning project that predicts whether a telecom customer is likely to churn.
The system is built using modular pipeline architecture, experiment tracking, and reproducible training.

📌 Problem Statement

Customer churn is a major business problem for subscription companies.
The goal of this project is to build a robust ML system that can predict:

Will a customer leave the service? (Yes / No)
This enables proactive retention strategies and revenue protection.

🧠 Solution Overview

This project implements a complete ML lifecycle:

- Data ingestion from MySQL
- Data validation & preprocessing
- Feature transformation
- Model training & hyperparameter tuning
- Experiment tracking with MLflow
- Best model selection
- Artifact persistence for deployment

🏗️ Project Architecture
MySQL → Data Ingestion → Data Transformation → Model Training
                                        ↓
                                   Preprocessor.pkl
                                        ↓
                                   model.pkl
                                        ↓
                                   MLflow / DagsHub

⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- MLflow
- DagsHub
- MySQL

📂 Project Structure
ML_PROJECT_2/
│
├── artifacts/                # saved outputs
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebook/                 # EDA & experiments
│
├── src/MLProject2/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── app.py                    # pipeline runner
├── requirements.txt
└── README.md

🤖 Models Compared

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
Hyperparameters tuned using GridSearchCV.

🏆 Final Result

Best Model: Gradient Boosting
Accuracy: ~80%

(Varies slightly per run due to randomness.)

📊 Experiment Tracking

All runs, parameters, and metrics are logged using MLflow and stored on DagsHub.
You can:

✔ compare models
✔ inspect metrics
✔ download artifacts
✔ reproduce experiments

▶️ How to Run:

1️⃣ Clone repository
git clone <repo-url>
cd ML_PROJECT_2

2️⃣ Create environment
pip install -r requirements.txt

3️⃣ Set environment variables (.env)
host=localhost
user=root
password=your_password
db=customer_churn_db

4️⃣ Run pipeline
python app.py

💾 Output

After run:

- trained model saved → artifacts/model.pkl
- preprocessor saved → artifacts/preprocessor.pkl
MLflow experiment logged

🎯 Key Highlights (Resume Points)

- Designed modular, reusable ML pipeline
- Implemented MySQL → ML training workflow
- Applied feature engineering & preprocessing
- Automated hyperparameter tuning
- Integrated MLflow for experiment tracking
- Enabled reproducibility & deployment readiness

🔮 Future Improvements

- Add model explainability (SHAP)
- Build prediction API
- CI/CD integration
- automated retraining
- monitoring


👤 Author:

Pulkit Chhabra
Aspiring Data Scientist | Machine Learning Enthusiast