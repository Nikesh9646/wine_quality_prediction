🍷 Wine Quality Prediction
📌 Project Overview

This project predicts the quality of wine using machine learning classifiers based on chemical properties of the wine. It demonstrates data preprocessing, visualization, model building, and evaluation in a step-by-step manner.

🔎 Step 1: Understand the Dataset

Dataset Used: WineQT.csv

Shape: 1143 rows × 13 columns

Target Variable: quality (integer 3–8)

Features:

fixed acidity

volatile acidity

citric acid

residual sugar

chlorides

free sulfur dioxide

total sulfur dioxide

density

pH

sulphates

alcohol

Note: Column Id is just an index and is removed during preprocessing.

🛠️ Step 2: Data Preprocessing

Dropped unnecessary columns (Id).

Checked for missing values (none found).

Split dataset into:

X (features)

y (target: quality)

Used train_test_split (80% training, 20% testing).

Scaled features with StandardScaler (important for SVM and SGD).

📊 Step 3: Exploratory Data Analysis (EDA)

Distribution Plot: Showed imbalance in wine quality ratings (most wines are rated 5–6).

Correlation Heatmap: Highlighted strong correlations (alcohol & quality positively correlated, volatile acidity negatively correlated).

Boxplots: Compared chemical features (e.g., alcohol, sulphates) across different wine qualities.

🤖 Step 4: Model Building

Trained three machine learning classifiers:

Random Forest Classifier (tree-based ensemble method).

Stochastic Gradient Descent (SGD) Classifier (linear model with gradient descent).

Support Vector Classifier (SVC) (max-margin classifier).

📈 Step 5: Model Evaluation

Metrics used:

Accuracy

Confusion Matrix

Precision, Recall, F1-score (from classification_report)

Results:

Random Forest gave the best accuracy and balanced predictions.

SGD and SVC struggled with rare quality classes (class imbalance issue).

🔑 Step 6: Feature Importance

Random Forest feature importance showed that:

Alcohol, sulphates, and citric acid are strong predictors of wine quality.

Volatile acidity has a negative impact on wine quality.

🚀 Step 7: Insights & Conclusion

Random Forest is the most reliable classifier for this dataset.

Alcohol level is the most significant feature influencing quality.

Rare wine ratings (e.g., 3 or 8) are difficult to classify due to limited samples.

🧩 Step 8: Future Improvements

Handle class imbalance using SMOTE or resampling.

Try XGBoost or LightGBM for better performance.

Deploy the model as a Streamlit/Flask web app.

⚙️ Technologies Used

Python

Pandas, NumPy

Matplotlib, Seaborn

scikit-learn

📂 Project Structure
Wine-Quality-Prediction/
│── WineQT.csv               # Dataset
│── wine_quality.ipynb       # Jupyter Notebook (full project)
│── README.md                # Documentation

👨‍💻 Author

Nikesh Penala

💼 Aspiring Data Analyst / ML Engineer

📧 [Your Email Here]

🌐 Open to collaborations in ML & Data Science
