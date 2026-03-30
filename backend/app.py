from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import get_prediction
import os

app = Flask(__name__)

# ✅ Allow only the deployed Vercel frontend
CORS(app, origins=["https://stock-frontend-lyart.vercel.app", "http://localhost:3000"])


@app.route("/")
def home():
    return "Stock Prediction API Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validate input
        if not data or "stock" not in data:
            return jsonify({"error": "Stock symbol required"}), 400

        stock = data.get("stock")

        # Call prediction function
        result = get_prediction(stock)

        return jsonify({
            "stock": stock,
            "current_price": result["current_price"],
            "predicted_price": result["predicted_price"],
            "change_percent": result["change_percent"]
        })

    except Exception as e:
        # Debugging support
        return jsonify({
            "error": "Server error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)