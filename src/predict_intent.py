import onnxruntime as ort
import json
import numpy as np
import os

# ========== Load label map ==========
with open("data/label_maps.json") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}

# ========== Emergency keyword rules ==========
emergency_keywords = [
    "fire", "smoke", "leak", "toxic", "fumes",
    "explosion", "blast", "suffocating", "can't breathe",
    "pressure dropping", "hull breach", "gas leak",
    "support failing", "danger", "emergency"
]

# ========== Load ONNX Model ==========
onnx_model = "models/intent_model.onnx"

session = ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

def preprocess(text):
    text = text.lower()
    tokens = text.split()

    # convert tokens into IDs using simple hashing
    # (this avoids using transformers)
    ids = [abs(hash(t)) % 30000 for t in tokens]

    # pad / crop to length 20
    max_len = 20
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return np.array([ids], dtype=np.int64)

def predict_intent(text):
    t = text.lower()

    # RULE-BASED HARD EMERGENCY CHECK
    for word in emergency_keywords:
        if word in t:
            return "emergency"

    X = preprocess(text)

    inputs = {session.get_inputs()[0].name: X}
    outputs = session.run(None, inputs)
    pred = np.argmax(outputs[0], axis=1)[0]

    return id_to_label[pred]

if __name__ == "__main__":
    while True:
        t = input("Enter: ")
        print("Intent:", predict_intent(t))
