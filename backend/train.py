import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def train_models():
    print("Loading data...")
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to diabetes.csv (one level up)
    csv_path = os.path.join(script_dir, '..', 'diabetes.csv')
    
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: diabetes.csv not found in parent directory.")
        return

    # 1. Data Cleaning
    # In this dataset, 0 values in these columns are physically impossible and represent missing data
    invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    print(f"Replacing 0s with NaN in {invalid_cols}...")
    df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

    print("Imputing missing values with Median...")
    # Using median is more robust to outliers than mean
    for col in invalid_cols:
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('median'))

    # 2. Feature Separation
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. Train-Test Split
    # Stratify ensures the test set has the same proportion of diabetic cases as the original data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. SMOTE (Synthetic Minority Over-sampling Technique)
    # This creates synthetic examples of diabetic cases to balance the training data
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original Train Shape: {y_train.value_counts().to_dict()}")
    print(f"Resampled Train Shape: {y_train_resampled.value_counts().to_dict()}")

    # 5. Scaling
    # Essential for SVM and Logistic Regression
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # 6. Model Training
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}

    print("\nTraining Models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train_resampled)
        
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred) # Crucial for health data!
        
        results[name] = {'model': model, 'accuracy': acc, 'recall': recall}
        
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 30)

    # 7. Save Artifacts
    print("\nSaving models and scaler...")
    joblib.dump(results['Logistic Regression']['model'], os.path.join(script_dir, 'model_lr.pkl'))
    joblib.dump(results['SVM']['model'], os.path.join(script_dir, 'model_svm.pkl'))
    joblib.dump(results['Random Forest']['model'], os.path.join(script_dir, 'model_rf.pkl'))
    joblib.dump(scaler, os.path.join(script_dir, 'scaler.pkl'))
    
    # Save metrics for the API to display
    metrics = {name: {'accuracy': info['accuracy'], 'recall': info['recall']} for name, info in results.items()}
    joblib.dump(metrics, os.path.join(script_dir, 'model_metrics.pkl'))

    # Save Feature Importance
    feature_importance = {
        'feature_names': X.columns.tolist(),
        'lr_coef': results['Logistic Regression']['model'].coef_[0].tolist(),
        'rf_importance': results['Random Forest']['model'].feature_importances_.tolist()
    }
    joblib.dump(feature_importance, os.path.join(script_dir, 'feature_importance.pkl'))
    
    print("Training complete. Artifacts saved.")

if __name__ == "__main__":
    train_models()
