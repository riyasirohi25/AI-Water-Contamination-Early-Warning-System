# 💧 Explainable AI-Based Early Warning Framework for Water Potability Prediction

<p align="center">
  <b>A Machine Learning and Explainable AI Framework for Predicting Water Potability and Identifying Key Water Quality Risk Factors</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-purple?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/SMOTE-Class%20Balancing-green?style=for-the-badge" alt="SMOTE">
  <img src="https://img.shields.io/badge/Research-Project-red?style=for-the-badge" alt="Research Project">
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#research-contributions">Contributions</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#results">Results</a> •
  <a href="#explainable-ai">Explainability</a> •
  <a href="#research-paper">Research Paper</a>
</p>

---

## 🔬 Research Project

This repository contains the implementation, experiments, analysis, and research artifacts for the study:

> **Explainable AI Framework for Water Potability Prediction Using Machine Learning**

The research investigates how machine learning can be used to predict water potability from physicochemical water-quality parameters while maintaining model interpretability through **Explainable AI (XAI)**.

The proposed framework combines **data preprocessing, class-imbalance correction using SMOTE, supervised machine learning, model evaluation, and SHAP-based explainability** to build an interpretable early-warning approach for water quality assessment.

---

## 📌 Overview

Access to safe drinking water remains an important public health and environmental challenge. Conventional water-quality assessment often depends on laboratory-based chemical analysis, which can be expensive and resource-intensive when applied at scale.

This research explores a data-driven alternative by using machine learning to classify water samples as **potable or non-potable** based on their physicochemical characteristics.

A major focus of the research is not only prediction accuracy, but also **understanding why the model makes a particular prediction**.

The framework therefore integrates **SHAP (SHapley Additive exPlanations)** to investigate feature contributions and identify the physicochemical parameters that have the greatest influence on water-potability predictions.

---

## 🎯 Research Objective

The primary objective of this research is to develop an **Explainable AI-based early warning framework for water potability prediction** that:

* Predicts whether a water sample is potable or non-potable.
* Handles missing values and feature-scale differences.
* Addresses severe class imbalance using **SMOTE**.
* Compares multiple supervised machine learning classifiers.
* Evaluates models using multiple performance metrics.
* Uses **SHAP** to explain model predictions.
* Identifies the physicochemical parameters most strongly associated with predicted water potability.
* Provides a foundation for future real-time and IoT-enabled water-quality monitoring systems.

---

## 📊 Dataset

The experiments use the publicly available **Water Potability** dataset containing:

* **3,276 water samples**
* **9 physicochemical parameters**
* **1 binary target variable**

### Input Features

| Feature           | Description                                  |
| ----------------- | -------------------------------------------- |
| `pH`              | Acidity/alkalinity of water                  |
| `Hardness`        | Concentration of calcium and magnesium salts |
| `TDS`             | Total Dissolved Solids                       |
| `Chloramines`     | Chloramine concentration                     |
| `Sulfate`         | Sulfate concentration                        |
| `Conductivity`    | Electrical conductivity of water             |
| `Organic Carbon`  | Organic carbon concentration                 |
| `Trihalomethanes` | Concentration of trihalomethanes             |
| `Turbidity`       | Cloudiness of water                          |

### Target

| Value | Meaning     |
| ----- | ----------- |
| `0`   | Non-potable |
| `1`   | Potable     |

The dataset contains a substantial class imbalance, with the potable class representing only a small portion of the original samples. This imbalance was explicitly addressed during model development using SMOTE.

---

# 🧠 Methodology

The proposed research framework follows an end-to-end machine learning pipeline:

```text
                Water Quality Dataset
                         │
                         ▼
              Exploratory Data Analysis
                         │
                         ▼
              Missing Value Imputation
                         │
                         ▼
                  Feature Scaling
                         │
                         ▼
                Train / Test Split
                         │
                         ▼
              SMOTE on Training Data
                         │
                         ▼
        ┌────────────────────────────────┐
        │       Machine Learning Models  │
        │                                │
        │  Logistic Regression           │
        │  Decision Tree                 │
        │  Random Forest                 │
        └────────────────────────────────┘
                         │
                         ▼
                 Model Evaluation
                         │
                         ▼
               Best Model Selection
                         │
                         ▼
                  SHAP Analysis
                         │
                         ▼
             Feature-Level Insights
```

---

## 1. 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

* Feature distributions
* Missing values
* Class distribution
* Feature relationships
* Correlations between physicochemical parameters
* Distribution of the target variable

The analysis revealed significant class imbalance in the original dataset, motivating the use of a dedicated imbalance-handling strategy.

---

## 2. 🧹 Data Preprocessing

### Missing Value Imputation

Missing values were handled using **column-wise median imputation**.

Median imputation was selected to provide robustness against potential outliers in the physicochemical measurements.

### Feature Scaling

Features were normalized using:

```python
StandardScaler
```

This transforms the features to approximately zero mean and unit variance.

---

## 3. ⚖️ Class Imbalance Handling

The original dataset contains a significant imbalance between potable and non-potable water samples.

To address this, the research uses:

### SMOTE — Synthetic Minority Oversampling Technique

SMOTE was applied **only to the training data**.

This is important because applying SMOTE before splitting the dataset can introduce synthetic information derived from test samples and lead to data leakage.

The workflow therefore follows:

```text
Original Dataset
       │
       ▼
Train / Test Split
       │
       ├──────────────► Test Set
       │
       ▼
  SMOTE on
 Training Set
       │
       ▼
Balanced Training Data
```

---

# 🤖 Machine Learning Models

Three supervised classification models were implemented and compared.

## Logistic Regression

Used as the baseline linear classifier.

It provides a useful reference point for evaluating whether more complex non-linear models provide significant improvements.

---

## Decision Tree

A non-linear rule-based classifier capable of learning decision boundaries through feature-based splits.

---

## Random Forest

An ensemble learning method consisting of multiple decision trees.

Random Forest was selected as the final model because it demonstrated the strongest overall predictive performance among the evaluated approaches.

---

# 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Five-Fold Stratified Cross-Validation

The dataset was divided into **80% training and 20% testing subsets** using stratified sampling.

---

# 🏆 Results

The experimental results demonstrate that Random Forest significantly outperformed the Logistic Regression baseline and Decision Tree classifier.

| Model               |   Accuracy |  Precision |     Recall |   F1-Score |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     78.28% |     23.51% |     82.21% |     36.57% |
| Decision Tree       |     99.53% |     96.79% |     97.05% |     96.92% |
| **Random Forest**   | **99.68%** | **96.97%** | **98.88%** | **97.92%** |

### Random Forest Performance

**Test Accuracy:** `99.68%`

**Precision:** `96.97%`

**Recall:** `98.88%`

**F1-Score:** `97.92%`

**Five-Fold Cross-Validation Accuracy:** `99.67%`

The Random Forest confusion matrix resulted in:

* **18,430 True Negatives**
* **1,506 True Positives**
* **47 False Positives**
* **17 False Negatives**

The relatively low number of false negatives is particularly relevant for a water-safety prediction setting where failing to identify a potentially unsafe classification can have important consequences.

---

# 🔍 Explainable AI

High predictive performance alone is not sufficient for many real-world decision-support applications.

This research therefore incorporates **SHAP (SHapley Additive exPlanations)** to understand the contribution of individual physicochemical parameters to model predictions.

SHAP was applied to the trained Random Forest model to provide:

### Global Explainability

Identifies which features have the greatest overall influence on the model.

### Instance-Level Explainability

Allows individual predictions to be analyzed based on the contribution of each input feature.

---

# 🧬 Feature Importance

The Random Forest feature-importance analysis produced the following ranking:

| Rank | Feature             | Importance |
| ---: | ------------------- | ---------: |
|    1 | **Conductivity**    | **22.24%** |
|    2 | **TDS**             | **18.40%** |
|    3 | **Trihalomethanes** | **14.62%** |
|    4 | Hardness            |     11.33% |
|    5 | Organic Carbon      |     10.85% |
|    6 | Sulfate             |      8.81% |
|    7 | Chloramines         |      5.48% |
|    8 | Turbidity           |      4.70% |
|    9 | pH                  |      3.57% |

The analysis indicates that **conductivity, total dissolved solids (TDS), and trihalomethanes** were among the most influential variables in the Random Forest predictions.

---

# 📊 SHAP Findings

The SHAP analysis provides additional insight into how the important features influence predictions.

The global SHAP analysis indicates that:

* **Conductivity** has a major influence on model output.
* **TDS** is another major contributor to water-potability predictions.
* **Trihalomethanes** also have substantial predictive influence.
* Higher values of conductivity and TDS were associated with a negative contribution toward predicted potability in the model.

This explainability layer helps bridge the gap between:

```text
Prediction
    +
Accuracy
    ↓
Interpretability
    ↓
Understanding Model Decisions
```

---

# 💡 Key Research Findings

### 1. Random Forest performed best

Random Forest achieved the highest overall performance among the three evaluated models with **99.68% accuracy**.

### 2. Class imbalance matters

The original water-potability dataset contains substantial class imbalance. SMOTE was therefore incorporated into the training pipeline to improve minority-class representation.

### 3. Explainability provides additional insight

SHAP analysis allowed the model's predictions to be interpreted rather than treating Random Forest as a complete black box.

### 4. Conductivity and TDS were dominant predictors

Feature-importance analysis identified **conductivity and TDS** as the two most influential features, followed by trihalomethanes.

### 5. Cross-validation supported model generalization

Five-fold stratified cross-validation achieved an average accuracy of **99.67%**, providing additional evidence of strong model performance on the evaluated dataset.

---

# 🧪 Experimental Setup

The framework was implemented in Python using standard scientific computing and machine learning libraries.

### Core Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Imbalanced-learn**
* **SHAP**

### Machine Learning

* Logistic Regression
* Decision Tree
* Random Forest
* SMOTE
* StandardScaler
* Stratified train/test split
* Five-fold cross-validation

### Explainable AI

* SHAP
* Random Forest feature importance
* SHAP summary / beeswarm analysis

---

# 📁 Repository Structure

```text
AI-Water-Contamination-Early-Warning-System/
│
├── data/
│   └── Dataset and data-related files
│
├── literature review/
│   └── Literature review and research references
│
├── notebooks/
│   └── Exploratory analysis and experimentation
│
├── paper/
│   ├── IEEE_Template.docx
│   ├── draft.docx
│   └── undertaking.docx
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explainability.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The repository separates the research workflow into preprocessing, model training, evaluation, and explainability components, while maintaining dedicated folders for the dataset, literature review, notebooks, and research-paper materials.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/riyasirohi25/AI-Water-Contamination-Early-Warning-System.git
```

Navigate to the project directory:

```bash
cd AI-Water-Contamination-Early-Warning-System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The source directory contains separate modules for the major stages of the research pipeline.

### Preprocessing

```bash
python src/preprocessing.py
```

### Model Training

```bash
python src/train.py
```

### Evaluation

```bash
python src/evaluate.py
```

### Explainability

```bash
python src/explainability.py
```

> The exact execution order and input/output paths should follow the configuration and implementation currently present in the repository.

---

# 📚 Research Workflow

The research process followed the following progression:

```text
Problem Identification
        ↓
Literature Review
        ↓
Dataset Selection
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Class Imbalance Analysis
        ↓
SMOTE-Based Balancing
        ↓
Model Development
        ↓
Comparative Evaluation
        ↓
Random Forest Selection
        ↓
Feature Importance Analysis
        ↓
SHAP Explainability
        ↓
Results & Discussion
        ↓
Research Paper
```

---

# 📄 Research Paper

### Title

**Explainable AI Framework for Water Potability Prediction Using Machine Learning**

### Author

**Riya Sirohi**

Department of Computer Science and Engineering
Mody University of Science and Technology, Jaipur, India

The research paper presents the complete methodology, experimental evaluation, feature-importance analysis, SHAP explainability analysis, limitations, and future research directions associated with this repository.

📄 **[View Research Paper / Paper Files](./paper/)**

---

# 🔬 Research Contributions

The research makes the following contributions:

1. **End-to-end machine learning framework** for binary water-potability classification.
2. **Systematic preprocessing pipeline** including missing-value imputation and feature normalization.
3. **SMOTE-based class imbalance handling** applied exclusively to training data.
4. **Comparative evaluation** of Logistic Regression, Decision Tree, and Random Forest.
5. **Five-fold stratified cross-validation** for evaluating model generalization.
6. **SHAP-based explainability** for interpreting Random Forest predictions.
7. Identification of **conductivity, TDS, and trihalomethanes** as major predictive factors.
8. Development of a research foundation for future **real-time and IoT-enabled water-quality monitoring**.

---

# ⚠️ Limitations

Although the framework achieved strong performance on the evaluated dataset, several limitations remain.

### Benchmark Dataset

The research uses a publicly available benchmark dataset rather than real-time water-quality streams.

### Temporal Variation

Temporal changes in water-quality characteristics were not incorporated into the current modeling framework.

### Geographic Variation

The current study does not explicitly model geographic differences between water sources and regions.

### Real-Time Monitoring

The current implementation does not represent a deployed IoT-based real-time monitoring system.

Therefore, the reported results should be interpreted within the scope of the evaluated benchmark dataset rather than as evidence of deployment-ready real-world performance.

---

# 🚀 Future Work

Future development of this research can focus on:

### 🌐 Real-Time Water Quality Monitoring

Integrating IoT sensors to collect continuous water-quality measurements.

```text
IoT Sensors
     ↓
Real-Time Data Stream
     ↓
Preprocessing
     ↓
ML Model
     ↓
SHAP Explainability
     ↓
Early Warning
```

### 📍 Geographically Diverse Datasets

Evaluating the framework across water sources and geographical regions to improve robustness.

### ⏱️ Temporal Modeling

Incorporating time-series data to identify changes and trends in water quality.

### 📊 Multi-Class Water Quality Assessment

Extending binary potable/non-potable classification into multiple water-quality grades.

### 🔔 Early Warning Integration

Developing a complete alerting layer that can notify users or authorities when predicted water quality crosses defined risk levels.

---

# 🛡️ Research Integrity

This project distinguishes between **experimental machine learning prediction** and **real-world water-quality certification**.

The model predicts water potability based on the features available in the benchmark dataset. It should not be considered a replacement for laboratory water-quality testing or regulatory certification.

---

# 📌 Why Explainability Matters

In environmental and public-health applications, a prediction alone may not be sufficient.

For example:

```text
Model Prediction
      │
      ▼
"Non-Potable"
      │
      ▼
Why?
      │
      ├── Conductivity
      ├── TDS
      ├── Trihalomethanes
      └── Other physicochemical parameters
```

The integration of SHAP provides a mechanism for investigating the factors contributing to model predictions, making the system more interpretable for research and potential decision-support applications.

---

# 📈 Performance Summary

```text
                     Random Forest

Accuracy              99.68%
Cross-Validation      99.67%
Precision              96.97%
Recall                 98.88%
F1-Score               97.92%

Top Predictors
────────────────────────────────────
1. Conductivity        22.24%
2. TDS                 18.40%
3. Trihalomethanes     14.62%
```

---

# 🧰 Tech Stack

| Category            | Technologies                                      |
| ------------------- | ------------------------------------------------- |
| Language            | Python                                            |
| Data Processing     | Pandas, NumPy                                     |
| Visualization       | Matplotlib, Seaborn                               |
| Machine Learning    | Scikit-learn                                      |
| Imbalanced Learning | Imbalanced-learn / SMOTE                          |
| Explainable AI      | SHAP                                              |
| Models              | Logistic Regression, Decision Tree, Random Forest |
| Research            | Literature Review, Experimental Evaluation        |
| Documentation       | IEEE Research Paper                               |
| Version Control     | Git / GitHub                                      |

---

# 📖 References

The research is supported by literature covering:

* Machine learning for water-quality prediction
* Water-potability classification
* Explainable AI for environmental applications
* SMOTE and class-imbalance handling
* Ensemble learning
* Water-quality forecasting
* Data-driven environmental monitoring

The complete references are available in the accompanying research paper.

---

# 👩‍💻 Author

## Riya Sirohi

**Computer Science & Engineering**
**Mody University of Science and Technology**

Research interests include:

* Artificial Intelligence
* Machine Learning
* Explainable AI
* Data Analytics
* Environmental AI
* Applied Machine Learning

---

# ⭐ Acknowledgement

This project was developed as an academic research project exploring the application of machine learning and Explainable AI to water-quality assessment.

The work combines experimental machine learning, statistical analysis, feature-importance analysis, and model explainability to investigate an interpretable approach to water-potability prediction.

---

# 📜 License

This repository is intended primarily for academic and research purposes.

If you use this work, please provide appropriate attribution to the original author and research paper.

---

<p align="center">
  <b>Explainable AI × Machine Learning × Environmental Intelligence</b>
  <br>
  <sub>Research-driven approaches toward intelligent water-quality monitoring</sub>
</p>
