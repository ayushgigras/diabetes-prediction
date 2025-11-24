from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import traceback
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
from flask import send_file

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load Artifacts
MODELS = {}
SCALER = None
METRICS = None
FEATURE_IMPORTANCE = None

def load_artifacts():
    global MODELS, SCALER, METRICS, FEATURE_IMPORTANCE
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        MODELS['Logistic Regression'] = joblib.load(os.path.join(base_path, 'model_lr.pkl'))
        MODELS['SVM'] = joblib.load(os.path.join(base_path, 'model_svm.pkl'))
        MODELS['Random Forest'] = joblib.load(os.path.join(base_path, 'model_rf.pkl'))
        SCALER = joblib.load(os.path.join(base_path, 'scaler.pkl'))
        METRICS = joblib.load(os.path.join(base_path, 'model_metrics.pkl'))
        FEATURE_IMPORTANCE = joblib.load(os.path.join(base_path, 'feature_importance.pkl'))
        print("[+] Artifacts loaded successfully.")
        return None
    except Exception as e:
        print(f"[-] Error loading artifacts: {e}")
        return str(e)

load_artifacts()

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/api/reload', methods=['GET'])
def reload_artifacts():
    """Force reload of artifacts."""
    error = load_artifacts()
    if FEATURE_IMPORTANCE:
        return jsonify({'status': 'Artifacts reloaded successfully', 'features_loaded': True})
    return jsonify({'status': 'Reload failed', 'features_loaded': False, 'error': error}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Return model performance metrics."""
    if METRICS:
        return jsonify(METRICS)
    return jsonify({'error': 'Metrics not loaded'}), 500

@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Generate and return a comparison graph."""
    if not METRICS:
        return jsonify({'error': 'Metrics not loaded'}), 500

    try:
        # Prepare data for plotting
        models = list(METRICS.keys())
        accuracy = [METRICS[m]['accuracy'] for m in models]
        recall = [METRICS[m]['recall'] for m in models]

        # Set style
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Create grouped bar chart
        x = np.arange(len(models))
        width = 0.35

        plt.bar(x - width/2, accuracy, width, label='Accuracy', color='#3b82f6') # Blue
        plt.bar(x + width/2, recall, width, label='Recall', color='#10b981')   # Green

        # Labels and Title
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, models)
        plt.ylim(0, 1.1)
        plt.legend()

        # Save to buffer
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph/features/<model_type>', methods=['GET'])
def get_feature_graph(model_type):
    """Generate feature importance graph for LR or RF."""
    if not FEATURE_IMPORTANCE:
        return jsonify({'error': 'Feature importance not loaded'}), 500

    try:
        features = FEATURE_IMPORTANCE['feature_names']
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        if model_type == 'lr':
            # Logistic Regression Coefficients
            values = FEATURE_IMPORTANCE['lr_coef']
            colors = ['#ef4444' if v > 0 else '#3b82f6' for v in values] # Red for positive risk, Blue for negative
            sns.barplot(x=values, y=features, palette=colors)
            plt.title('Logistic Regression Feature Coefficients')
            plt.xlabel('Coefficient Value')
            
        elif model_type == 'rf':
            # Random Forest Importance
            values = FEATURE_IMPORTANCE['rf_importance']
            # Sort for better visualization
            sorted_idx = np.argsort(values)
            sorted_features = [features[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            
            sns.barplot(x=sorted_values, y=sorted_features, color='#10b981') # Green
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance Score')
            
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        plt.tight_layout()

        # Save to buffer
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Extract and validate features
        features = []
        for feature in FEATURE_NAMES:
            val = data.get(feature)
            if val is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            try:
                features.append(float(val))
            except ValueError:
                return jsonify({'error': f'Invalid value for {feature}'}), 400

        # Preprocess
        X = np.array(features).reshape(1, -1)
        if SCALER:
            X_scaled = SCALER.transform(X)
        else:
            return jsonify({'error': 'Scaler not loaded'}), 500

        # Get Predictions from all models
        results = {}
        total_prob = 0
        
        for name, model in MODELS.items():
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_scaled)[0][1]
            else:
                prob = float(model.predict(X_scaled)[0])
            
            results[name] = {
                'prediction': 'Diabetic' if prob >= 0.5 else 'Non-Diabetic',
                'probability': float(prob)
            }
            total_prob += prob

        # Ensemble Logic (Average Probability)
        avg_prob = total_prob / len(MODELS)
        
        # Risk Assessment
        if avg_prob > 0.7:
            risk_level = 'High'
            recommendation = 'Please consult a doctor immediately.'
        elif avg_prob > 0.35:
            risk_level = 'Moderate'
            recommendation = 'Monitor your health and maintain a balanced diet.'
        else:
            risk_level = 'Low'
            recommendation = 'Keep up the healthy lifestyle!'

        response = {
            'ensemble_probability': float(avg_prob),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'model_details': results
        }

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
