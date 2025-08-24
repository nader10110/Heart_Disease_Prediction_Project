# ❤️ Comprehensive Machine Learning Full Pipeline on Heart Disease (UCI Dataset)

## 📌 General Description
This project analyzes, predicts, and visualizes **heart disease risks** using **Machine Learning**.  
It covers the full pipeline: **data preprocessing, feature selection, dimensionality reduction (PCA), supervised & unsupervised learning, hyperparameter tuning, and deployment**.  

A **Streamlit UI** is developed for real-time interaction and all work is hosted here on GitHub.

---

## 🎯 Objectives
- ✅ Perform **Data Preprocessing & Cleaning** (missing values, encoding, scaling).  
- ✅ Apply **Dimensionality Reduction (PCA)** to retain essential features.  
- ✅ Implement **Feature Selection** (statistical & ML-based methods).  
- ✅ Train **Classification Models**: Logistic Regression, Decision Trees, Random Forest, SVM.  
- ✅ Apply **Unsupervised Learning**: K-Means & Hierarchical Clustering.  
- ✅ Perform **Hyperparameter Tuning** with GridSearchCV & RandomizedSearchCV.  
- ✅ Build a **Streamlit UI** for user interaction.  

---

## 🛠 Tools & Technologies
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, (TensorFlow/Keras optional)  
- **Dimensionality Reduction & Feature Selection:** PCA, RFE, Chi-Square Test  
- **Supervised Models:** Logistic Regression, Decision Tree, Random Forest, SVM  
- **Unsupervised Models:** K-Means, Hierarchical Clustering  
- **Model Optimization:** GridSearchCV, RandomizedSearchCV  
- **Deployment:** Streamlit, GitHub  

---


---

## 🔬 Workflow

### 1️⃣ Data Preprocessing
- Handle missing values (imputation/removal)  
- Encode categorical features (One-Hot Encoding)  
- Standardize numerical features (MinMaxScaler/StandardScaler)  
- Exploratory Data Analysis (EDA): histograms, heatmaps, boxplots  

**Deliverable:** Clean dataset ready for modeling ✅  

---

### 2️⃣ Dimensionality Reduction - PCA
- Apply PCA to reduce dimensionality while retaining variance  
- Determine number of components using **explained variance ratio**  
- Visualize results (scatter plot, cumulative variance plot)  

**Deliverable:** PCA-transformed dataset + variance plots ✅  

---

### 3️⃣ Feature Selection
- Random Forest / XGBoost feature importance  
- Recursive Feature Elimination (RFE)  
- Chi-Square Test  

**Deliverable:** Reduced dataset with selected features + feature importance plots ✅  

---

### 4️⃣ Supervised Learning (Classification)
- Train/Test split (80/20)  
- Models: Logistic Regression, Decision Tree, Random Forest, SVM  
- Metrics: Accuracy, Precision, Recall, F1-score, ROC, AUC  

**Deliverable:** Trained classification models + evaluation metrics ✅  

---

### 5️⃣ Unsupervised Learning (Clustering)
- K-Means (elbow method for optimal K)  
- Hierarchical Clustering (dendrogram analysis)  

**Deliverable:** Clustering results with visualizations ✅  

---

### 6️⃣ Hyperparameter Tuning
- Apply **GridSearchCV** and **RandomizedSearchCV**  
- Compare optimized vs baseline models  

**Deliverable:** Best performing model ✅  

---

### 7️⃣ Model Export & Deployment
- Save trained model with **joblib/pickle (.pkl)**  
- Build a **Streamlit UI** for user interaction  

**Deliverable:**  
- `final_model.pkl`  
- Functional Streamlit app ✅  


