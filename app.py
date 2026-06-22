import os
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

HISTORY_FILE = "history.json"

# ✅ Matches your notebook exactly
IMG_SIZE = (150, 150)

print("Loading model...")
model = load_model("model/cats_vs_dogs_model.keras")
print("Model loaded.")


# ── helpers ───────────────────────────────────────────────────────────────────

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # ✅ NO preprocess_input — your model has Rescaling(1./255) built in
    # ✅ Class order: alphabetical → cats=0, dogs=1
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        label, confidence = "Dog", float(prediction)
    else:
        label, confidence = "Cat", float(1 - prediction)

    return label, confidence


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE) as f:
        return json.load(f)


def save_to_history(label, confidence, img_path):
    history = load_history()
    history.insert(0, {
        "id": str(uuid.uuid4())[:8],
        "label": label,
        "confidence": f"{confidence:.1%}",
        "confidence_raw": round(confidence * 100, 1),
        "image": img_path.replace("\\", "/"),
        "date": datetime.now().strftime("%b %d, %Y · %H:%M"),
    })
    history = history[:50]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_path = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            ext = os.path.splitext(file.filename)[1]
            filename = f"{uuid.uuid4().hex[:8]}{ext}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            result = {
                "label": label,
                "confidence": f"{confidence:.1%}",
                "confidence_raw": round(confidence * 100, 1),
            }
            image_path = filepath.replace("\\", "/")
            save_to_history(label, confidence, image_path)
    return render_template("index.html", result=result, image_path=image_path)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
def history():
    records = load_history()
    return render_template("history.html", records=records)


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/clear-history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True)