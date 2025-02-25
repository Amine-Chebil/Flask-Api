from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import os

app = Flask(__name__)

# Allow CORS for specific origins (localhost + Render)
CORS(app, origins=["https://localhost:3000", "https://flask-api-p63c.onrender.com"])

# Load RoBERTa model for text classification from Hugging Face
classifier = pipeline(
    "zero-shot-classification", 
    model="FacebookAI/roberta-large-mnli"  # Load the correct model
)

# Define categories
categories = ["Gym or Fitness Center", "Spa", "Restaurant"]

@app.route("/classify", methods=["POST"])
def classify_email():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")
    text = f"Email Subject: {subject}. Email Body: {body}"

    # Perform classification
    result = classifier(text, categories, multi_label=True)

    # Print full output for debugging
    print("Model Output:", result)

    # Adjust threshold to include multiple relevant categories
    threshold = 0.3  # Lowering it to capture more categories
    classified_labels = [label for label, score in zip(result["labels"], result["scores"]) if score > threshold]

    return jsonify({"category": classified_labels if classified_labels else ["Uncategorized"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
