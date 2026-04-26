# Algerian Forest Fire Predictor 🔥

<div align="center">

```
███████╗ ██████╗ ██████╗ ███████╗███████╗████████╗    ███████╗██╗██████╗ ███████╗
██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██║██╔══██╗██╔════╝
█████╗  ██║   ██║██████╔╝█████╗  ███████╗   ██║       █████╗  ██║██████╔╝█████╗  
██╔══╝  ██║   ██║██╔══██╗██╔══╝  ╚════██║   ██║       ██╔══╝  ██║██╔══██╗██╔══╝  
██║     ╚██████╔╝██║  ██║███████╗███████║   ██║       ██║     ██║██║  ██║███████╗
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝       ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝
```

### End-to-End ML Regression Pipeline — From Raw Data to Cloud Deployment

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Ridge](https://img.shields.io/badge/Model-Ridge_Regression-8A2BE2?style=for-the-badge)](.)
[![Status](https://img.shields.io/badge/Status-Deployed-22C55E?style=for-the-badge)](.)

<br/>

> **EDA · Feature Selection · Regularization · StandardScaler · FastAPI · AWS Deployment**

</div>

---

## What is This Project?

A complete machine learning regression pipeline that predicts the **Fire Weather Index (FWI)** — a composite score used by meteorologists to assess wildfire danger — using the **Algerian Forest Fire Dataset**.

The project covers the full ML lifecycle:

- ✅ **Data Cleaning** — handling nulls, fixing dtypes, stripping whitespace, regional encoding
- ✅ **EDA** — correlation heatmaps, pairplots, boxplots, class distribution analysis
- ✅ **Feature Selection** — correlation-threshold-based multicollinearity removal
- ✅ **Regularization Benchmarking** — Linear, Lasso, Ridge, ElasticNet, LassoCV, RidgeCV compared
- ✅ **Deployment** — FastAPI REST API with HTML frontend, deployed on AWS

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        RAW DATASET                           │
│         Algerian Forest Fire (2 Regions · 244 rows)          │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      DATA CLEANING                           │
│   Null removal · dtype fixes · region encoding · dedup       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   EXPLORATORY DATA ANALYSIS                  │
│     Heatmap · Pairplot · Boxplot · Class distribution        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    FEATURE SELECTION                         │
│         Correlation threshold > 0.86 → drop features        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  FEATURE SCALING                             │
│              StandardScaler (fit on train only)              │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               REGULARIZATION BENCHMARKING                    │
│    Linear · Lasso · Ridge · ElasticNet · LassoCV · RidgeCV  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              BEST MODEL — Ridge Regression                   │
│              Pickled → ridge.pkl + scaler.pkl                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                FastAPI REST API + HTML Frontend              │
│                     Deployed on AWS                          │
└──────────────────────────────────────────────────────────────┘
```

---

## EDA Highlights

### Correlation Heatmap
Used to identify multicollinear features. Custom threshold function drops any feature pair with correlation > 0.86, preventing redundant signals from inflating model confidence.

![Correlation Heatmap](https://github.com/user-attachments/assets/333ec972-cd25-47df-9806-d745829c64b6)

---

## Feature Engineering

### Effect of StandardScaler
Standardizing features is critical for regularized models like Ridge — without it, features with larger scales dominate the penalty term. The comparison below shows the distribution shift before and after scaling.

![Before and After StandardScaler](https://github.com/user-attachments/assets/da0dfee3-31ae-4bfe-99e2-e5e9d19e2d02)

---

## Model Selection

Six regression models were benchmarked on the same train/test split:

| Model | Notes |
|---|---|
| Linear Regression | Baseline — no regularization |
| Lasso (L1) | Sparse feature selection via zero coefficients |
| **Ridge (L2)** | **Best MAE + R² — selected for deployment** |
| ElasticNet | L1 + L2 hybrid |
| LassoCV | Cross-validated alpha tuning across 100 values |
| RidgeCV | Cross-validated Ridge |

Ridge regression was selected based on lowest MAE and highest R² score on the test set.

### Ridge Regression — Predicted vs Actual (y_test vs y_pred)
A tight linear scatter indicates the model generalizes well with minimal bias.

![Ridge Regression Scatter Plot](https://github.com/user-attachments/assets/7f4e59ba-0a15-48ef-bfe2-da84e662da4d)

---

## Input Features

| Feature | Description |
|---|---|
| `Temperature` | Ambient temperature (°C) |
| `RH` | Relative Humidity (%) |
| `Ws` | Wind speed (km/h) |
| `Rain` | Rainfall (mm) |
| `FFMC` | Fine Fuel Moisture Code |
| `DMC` | Duff Moisture Code |
| `ISI` | Initial Spread Index |
| `Classes` | Fire / Not Fire (encoded) |
| `Region` | Bejaia (0) / Sidi Bel-Abbes (1) |

**Target:** `FWI` — Fire Weather Index

---

## Deployment

### FastAPI — REST API
Two endpoints power the application:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/predict` | Prediction form |
| `POST` | `/predict` | Accepts form input → returns FWI prediction |

![FastAPI Docs — Endpoints & Schemas](https://github.com/user-attachments/assets/d3697c73-4cf5-47ca-aef1-aef7e76f8af7)

### HTML Frontend
A lightweight form interface that accepts all 9 input features and displays the predicted FWI score instantly.

![Prediction Form UI](https://github.com/user-attachments/assets/94272237-daae-4a35-84d0-4d540f23733d)

---

## AWS Deployment

### Architecture
```
GitHub (main branch)
        │
        ▼  auto-trigger on push
AWS CodePipeline
        │
        ├── Stage 1: Source  (GitHub via GitHub App)
        │
        └── Stage 2: Deploy  (AWS Elastic Beanstalk)
                    │
                    ▼
        EC2 Instance (Python 3.13 · Amazon Linux 2023)
        Nginx → Uvicorn → FastAPI
```

### CodePipeline — Source + Deploy (Both Succeeded ✅)
Fully automated CI/CD — every push to GitHub triggers the pipeline and redeploys to EBS automatically. Zero manual steps.

<img width="1915" height="769" alt="Screenshot 2026-04-26 172233" src="https://github.com/user-attachments/assets/77ebc4af-e007-4bf2-92a2-eddc011a1ae7" />

### Elastic Beanstalk — Environment Health: Ok ✅
Environment `Algerian-Forest-Fire-Predictor-env` running on Python 3.13, WebServer tier, health status **Ok**.

<img width="1918" height="768" alt="Screenshot 2026-04-26 164807" src="https://github.com/user-attachments/assets/4e7b8d18-6561-488b-8698-f5c3e08240ed" />

### Live — FWI Prediction on AWS
App live on EBS public URL (`ap-south-1`), accepting real inputs and returning predictions via the deployed Ridge model.

<img width="875" height="801" alt="Screenshot 2026-04-26 172157" src="https://github.com/user-attachments/assets/604d9168-fe95-4282-8baf-10e21730ac99" />

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **ML** | Scikit-Learn | Preprocessing, modeling, evaluation |
| **Backend** | FastAPI | REST API + form handling |
| **Frontend** | HTML / Jinja2 | Prediction interface |
| **Serialization** | Pickle | Model + scaler persistence |
| **Cloud** | AWS | Production deployment |
| **EDA** | Matplotlib, Seaborn | Visualization |
| **Data** | Pandas, NumPy | Cleaning and transformation |

---

## Project Structure

```
Algerian-Forest-Fire-Predictor/
│
├── notebooks/
│   ├── EDA.ipynb                  # Data cleaning + exploratory analysis
│   └── Model_Training.ipynb       # Feature selection + model benchmarking
│
├── models/
│   ├── ridge.pkl                  # Trained Ridge Regression model
│   └── scaler.pkl                 # Fitted StandardScaler
│
├── templates/
│   ├── index.html                 # Landing page
│   └── home.html                  # Prediction form + result display
│
├── data/
│   ├── algerian_forest_fire.csv          # Raw dataset
│   └── Algerian_forest_fire_cleaned.csv  # Cleaned dataset
│
├── main.py                        # FastAPI application
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/ayushhh026/Algerian-Forest-Fire-Predictor.git
cd Algerian-Forest-Fire-Predictor
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn main:app --host localhost --port 8000 --reload
```

### 5. Open in browser
```
http://localhost:8000
```

---

## Key Engineering Decisions

**Why Ridge over Lasso?**
The dataset has correlated features (even after threshold filtering). Lasso tends to arbitrarily zero out one of a correlated pair, which can drop meaningful predictors. Ridge shrinks all coefficients proportionally, which is more stable for this dataset structure.

**Why StandardScaler fit only on train?**
Fitting the scaler on the full dataset would leak test distribution information into training — a subtle but real form of data leakage. The scaler is fit exclusively on `X_train` and then applied to `X_test`.

**Why custom correlation function over VIF?**
Variance Inflation Factor is computationally heavier and harder to threshold cleanly for automation. The custom correlation matrix approach gives direct control over the cutoff (0.86) and produces a reproducible, inspectable feature set.

---

## Roadmap

- [ ] Add cross-validation scores to model comparison table
- [ ] Interactive dashboard (Streamlit / Plotly)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Docker containerization
- [ ] Auto-retrain on new data uploads

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## Author

**Ayush Shetty**  
AI & Data Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-ayushhh026-181717?style=flat-square&logo=github)](https://github.com/ayushhh026)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Shetty-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ayush-shetty-830a03281/)

---

<div align="center">

**⭐ Star this repo if it helped you — it keeps the project alive.**

*Built end-to-end — data to deployment, zero shortcuts.*

</div>

---

> **Disclaimer:** FWI predictions are for educational purposes only. Do not use for real wildfire risk assessment.<img width="1915" height="769" alt="Screenshot 2026-04-26 172233" src="https://github.com/user-attachments/assets/7e05a263-3745-49eb-9d5f-f1f6135321b9" />
