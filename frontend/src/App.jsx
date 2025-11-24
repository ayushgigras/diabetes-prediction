import React, { useState, useEffect } from 'react';
import { Activity, AlertCircle, CheckCircle, HeartPulse, Info } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [formData, setFormData] = useState({
    Pregnancies: '2',
    Glucose: '120',
    BloodPressure: '70',
    SkinThickness: '20',
    Insulin: '79',
    BMI: '32',
    DiabetesPedigreeFunction: '0.47',
    Age: '33'
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // Fetch model metrics on load
    fetch(`${API_URL}/api/metrics`)
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error("Failed to load metrics", err));
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Something went wrong');
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const [activeTab, setActiveTab] = useState('prediction');
  const [graphView, setGraphView] = useState('performance');

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans">
      {/* Header */}
      <header className="bg-blue-600 text-white p-6 shadow-lg">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <HeartPulse size={32} />
          <h1 className="text-3xl font-bold">Diabetes Prediction System</h1>
        </div>
        <p className="max-w-4xl mx-auto mt-2 text-blue-100">
          Advanced AI-powered analysis using Logistic Regression, SVM, and Random Forest.
        </p>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-4xl mx-auto mt-6 px-6 border-b border-gray-200 flex gap-8">
        <button
          onClick={() => setActiveTab('prediction')}
          className={`pb-3 font-medium transition-colors ${activeTab === 'prediction'
            ? 'text-blue-600 border-b-2 border-blue-600'
            : 'text-gray-500 hover:text-gray-700'
            }`}
        >
          Prediction Tool
        </button>
        <button
          onClick={() => setActiveTab('methodology')}
          className={`pb-3 font-medium transition-colors ${activeTab === 'methodology'
            ? 'text-blue-600 border-b-2 border-blue-600'
            : 'text-gray-500 hover:text-gray-700'
            }`}
        >
          Analysis & Methodology
        </button>
      </div>

      <main className="max-w-4xl mx-auto p-6">

        {/* PREDICTION TAB */}
        {activeTab === 'prediction' && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Left Column: Input Form */}
            <div className="md:col-span-2 space-y-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                  <Activity className="text-blue-500" /> Patient Data
                </h2>

                <form onSubmit={handleSubmit} className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {Object.keys(formData).map((key) => (
                    <div key={key}>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {key.replace(/([A-Z])/g, ' $1').trim()}
                      </label>
                      <input
                        type="number"
                        step="any"
                        name={key}
                        value={formData[key]}
                        onChange={handleChange}
                        required
                        className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all outline-none"
                        placeholder={`Enter ${key}`}
                      />
                    </div>
                  ))}

                  <div className="sm:col-span-2 mt-4">
                    <button
                      type="submit"
                      disabled={loading}
                      className={`w-full py-3 px-6 rounded-lg text-white font-semibold shadow-md transition-all
                        ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg'}
                      `}
                    >
                      {loading ? 'Analyzing...' : 'Predict Risk'}
                    </button>
                  </div>
                </form>

                {error && (
                  <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center gap-2">
                    <AlertCircle size={20} />
                    {error}
                  </div>
                )}
              </div>

              {/* Results Section */}
              {result && (
                <div className="bg-white p-6 rounded-xl shadow-md border-l-4 border-blue-500 animate-fade-in">
                  <h2 className="text-2xl font-bold mb-4">Analysis Result</h2>

                  <div className="flex items-center gap-4 mb-6">
                    <div className={`p-4 rounded-full ${result.risk_level === 'High' ? 'bg-red-100 text-red-600' : result.risk_level === 'Moderate' ? 'bg-yellow-100 text-yellow-600' : 'bg-green-100 text-green-600'}`}>
                      {result.risk_level === 'High' ? <AlertCircle size={40} /> : <CheckCircle size={40} />}
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 uppercase tracking-wide">Risk Level</p>
                      <p className={`text-3xl font-bold ${result.risk_level === 'High' ? 'text-red-600' : result.risk_level === 'Moderate' ? 'text-yellow-600' : 'text-green-600'}`}>
                        {result.risk_level}
                      </p>
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg mb-6">
                    <h3 className="font-semibold mb-2">Recommendation:</h3>
                    <p className="text-gray-700">{result.recommendation}</p>
                  </div>

                  <h3 className="font-semibold mb-3 text-gray-700">Model Consensus:</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {Object.entries(result.model_details).map(([model, details]) => (
                      <div key={model} className="bg-gray-100 p-3 rounded-lg text-center">
                        <p className="text-xs text-gray-500 mb-1">{model}</p>
                        <p className={`font-bold ${details.prediction === 'Diabetic' ? 'text-red-500' : 'text-green-500'}`}>
                          {details.prediction}
                        </p>
                        <p className="text-xs text-gray-400">{(details.probability * 100).toFixed(1)}%</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Quick Info */}
            <div className="space-y-6">
              <div className="bg-blue-50 p-6 rounded-xl border border-blue-100">
                <h3 className="font-semibold text-blue-800 mb-2">How it works</h3>
                <p className="text-sm text-blue-700 mb-4">
                  Enter the patient's health metrics. Our system runs 3 different AI models to assess the risk of diabetes.
                </p>
                <button
                  onClick={() => setActiveTab('methodology')}
                  className="text-sm text-blue-600 font-semibold hover:underline"
                >
                  View Methodology &rarr;
                </button>
              </div>
            </div>
          </div>
        )}

        {/* METHODOLOGY TAB */}
        {activeTab === 'methodology' && (
          <div className="space-y-8 animate-fade-in">

            {/* Graph Section */}
            <section className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <Activity className="text-blue-500" /> Analysis Graphs
                </h2>

                {/* Graph Controls */}
                <div className="flex bg-gray-100 p-1 rounded-lg">
                  <button
                    onClick={() => setGraphView('performance')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${graphView === 'performance' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-500 hover:text-gray-700'
                      }`}
                  >
                    Performance
                  </button>
                  <button
                    onClick={() => setGraphView('lr')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${graphView === 'lr' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-500 hover:text-gray-700'
                      }`}
                  >
                    LR Features
                  </button>
                  <button
                    onClick={() => setGraphView('rf')}
                    className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${graphView === 'rf' ? 'bg-white shadow-sm text-blue-600' : 'text-gray-500 hover:text-gray-700'
                      }`}
                  >
                    RF Features
                  </button>
                </div>
              </div>

              <div className="flex flex-col md:flex-row gap-8 items-start">
                <div className="w-full md:w-2/3">
                  <img
                    src={
                      graphView === 'performance' ? `${API_URL}/api/graph?t=${Date.now()}` :
                        graphView === 'lr' ? `${API_URL}/api/graph/features/lr?t=${Date.now()}` :
                          `${API_URL}/api/graph/features/rf?t=${Date.now()}`
                    }
                    alt="Analysis Graph"
                    className="w-full rounded-lg shadow-sm border border-gray-200"
                  />
                </div>
                <div className="w-full md:w-1/3 space-y-4">
                  {graphView === 'performance' && (
                    <>
                      <div className="bg-blue-50 p-4 rounded-lg">
                        <h3 className="font-semibold text-blue-800 mb-1">Why Random Forest?</h3>
                        <p className="text-sm text-blue-700">
                          It achieved the highest accuracy (86%) by creating multiple decision trees, making it robust against overfitting.
                        </p>
                      </div>
                      <div className="bg-green-50 p-4 rounded-lg">
                        <h3 className="font-semibold text-green-800 mb-1">Why SVM?</h3>
                        <p className="text-sm text-green-700">
                          It had the best Recall (85%), meaning it's the safest model for catching positive cases.
                        </p>
                      </div>
                    </>
                  )}
                  {graphView === 'lr' && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="font-semibold text-gray-800 mb-1">Logistic Regression</h3>
                      <p className="text-sm text-gray-600">
                        This chart shows how each feature affects the risk score.
                        <br /><br />
                        All features shown here have a positive correlation with diabetes risk in this model.
                      </p>
                    </div>
                  )}
                  {graphView === 'rf' && (
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="font-semibold text-gray-800 mb-1">Random Forest</h3>
                      <p className="text-sm text-gray-600">
                        This chart shows which features are the most important for the model's decision making.
                        <br /><br />
                        Features with longer bars have a greater impact on the final prediction.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Methodology Details */}
            <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="font-bold text-lg mb-3">1. Data Cleaning</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  The dataset contained invalid '0' values for Glucose, BP, and BMI.
                  We replaced these zeros with the <strong>Median</strong> value of their respective classes.
                  This prevents the models from learning incorrect patterns from missing data.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="font-bold text-lg mb-3">2. Handling Imbalance (SMOTE)</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  The original data was 65% Non-Diabetic. To prevent bias, we used
                  <strong> SMOTE (Synthetic Minority Over-sampling Technique)</strong>.
                  It creates synthetic examples of diabetic cases, teaching the model to recognize them better.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="font-bold text-lg mb-3">3. Model Selection</h3>
                <ul className="list-disc list-inside text-gray-600 text-sm space-y-2">
                  <li><strong>Logistic Regression:</strong> Used as a baseline for linear relationships.</li>
                  <li><strong>SVM (RBF Kernel):</strong> Good for finding complex boundaries in high-dimensional data.</li>
                  <li><strong>Random Forest:</strong> An ensemble method that handles non-linear data very well.</li>
                </ul>
              </div>

              <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 className="font-bold text-lg mb-3">4. Why Recall Matters?</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  In medical AI, a <strong>False Negative</strong> (telling a sick person they are healthy) is the worst outcome.
                  We optimized our models to maximize <strong>Recall</strong> (Sensitivity), ensuring we catch as many potential cases as possible.
                </p>
              </div>
            </section>

          </div>
        )}

      </main>
    </div>
  );
}

export default App;
