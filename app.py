from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io

app = Flask(__name__)


model    = tf.keras.models.load_model("butterfly_cnn.keras")
classes  = np.load("label_classes.npy", allow_pickle=True)
IMG_SIZE = (150, 150)

def prepare_image(file_bytes):
    img       = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img       = img.resize(IMG_SIZE)
    arr       = np.array(img) / 255.0
    arr       = np.expand_dims(arr, axis=0)   # (1, 150, 150, 3)
    return arr

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file      = request.files["file"]
    img_array = prepare_image(file.read())

    predictions     = model.predict(img_array)
    predicted_idx   = int(np.argmax(predictions[0]))
    predicted_label = classes[predicted_idx]
    confidence      = float(predictions[0][predicted_idx]) * 100

    # Top 5
    top5_idx = np.argsort(predictions[0])[::-1][:5]
    top5 = [
        {"label": classes[i], "probability": round(float(predictions[0][i]) * 100, 2)}
        for i in top5_idx
    ]

    return jsonify({
        "prediction": predicted_label,
        "confidence": round(confidence, 2),
        "top5": top5
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
