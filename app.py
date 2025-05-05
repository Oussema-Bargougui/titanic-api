from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("titanic_model.pkl")

# Secret API key (used for raw API access)
API_KEY = "123"

# Home page message
@app.route('/')
def home():
    return "🎯 Titanic Prediction API Ready! POST to /predict or use /form to try it visually."

# ✅ Simple browser form
@app.route('/form', methods=['GET', 'POST'])
def form():
    result = None
    if request.method == 'POST':
        try:
            pclass = int(request.form['pclass'])
            sex = int(request.form['sex'])  # 1 = male, 0 = female
            age = float(request.form['age'])
            fare = float(request.form['fare'])

            features = [pclass, sex, age, fare]
            pred = model.predict([features])[0]
            result = f"Prediction: {'Survived ✅' if pred == 1 else 'Did not survive ❌'}"
        except Exception as e:
            result = f"Error: {str(e)}"

    # HTML template with form
    html = """
    <h2>🚢 Titanic Survival Prediction</h2>
    <form method="post">
        <label>Pclass (1, 2, 3):</label><br>
        <input type="number" name="pclass" required><br><br>

        <label>Sex (1 = Male, 0 = Female):</label><br>
        <input type="number" name="sex" required><br><br>

        <label>Age:</label><br>
        <input type="number" name="age" step="any" required><br><br>

        <label>Fare:</label><br>
        <input type="number" name="fare" step="any" required><br><br>

        <input type="submit" value="Predict">
    </form>

    {% if result %}
        <h3>{{ result }}</h3>
    {% endif %}
    """
    return render_template_string(html, result=result)

# Raw JSON API (still protected with API key)
@app.route('/predict', methods=['POST'])
def predict():
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        return jsonify({"error": "❌ Unauthorized"}), 401

    data = request.get_json()
    features = data.get("features")

    if not features or len(features) != 4:
        return jsonify({"error": "Expected 4 features: [Pclass, Sex, Age, Fare]"}), 400

    try:
        pred = model.predict([features])[0]
        return jsonify({
            "prediction": int(pred),
            "survived": "Yes" if pred == 1 else "No"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
