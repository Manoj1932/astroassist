import os
import json
import onnxruntime as ort
import numpy as np

# ----------------------------------------------------
# Paths (safe on Render + local)
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

label_map_path = os.path.join(MODEL_DIR, "label_maps.json")
onnx_model_path = os.path.join(MODEL_DIR, "intent_classifier.onnx")

# ----------------------------------------------------
# Load label map
# ----------------------------------------------------
with open(label_map_path, "r") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}

# ----------------------------------------------------
# Emergency rule keywords
# ----------------------------------------------------
emergency_keywords = [
    "fire", "smoke", "leak", "toxic", "fumes",
    "explosion", "blast", "suffocating", "can't breathe",
    "pressure dropping", "hull breach", "gas leak",
    "support failing", "danger", "emergency"
]

# ----------------------------------------------------
# ONNX Runtime Session (safer)
# ----------------------------------------------------
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False

session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"],
    sess_options=sess_options
)

# ----------------------------------------------------
# Stable Hash Function (Python hash() is unstable)
# ----------------------------------------------------
def stable_hash(text):
    """Deterministic 32-bit hash for consistent ONNX input."""
    return abs(np.int32(np.sum([ord(c) * (i + 1) for i, c in enumerate(text)])))

# ----------------------------------------------------
# Preprocess text for ONNX model
# ----------------------------------------------------
def preprocess(text):
    text = text.lower()
    tokens = text.split()

    # hashed token IDs (stable)
    ids = [stable_hash(t) % 30000 for t in tokens]

    # Pad / crop to fixed length
    max_len = 20
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return np.array([ids], dtype=np.int64)

# ----------------------------------------------------
# Predict intent
# ----------------------------------------------------
def predict_intent(text):

    t = text.lower()

    # 1. Emergency keyword rule
    for word in emergency_keywords:
        if word in t:
            return "emergency"

    # 2. Preprocess text → convert to ids
    X = preprocess(text)

    # 3. ONNX forward pass
    inputs = {session.get_inputs()[0].name: X}
    outputs = session.run(None, inputs)
    pred = int(np.argmax(outputs[0], axis=1)[0])

    return id_to_label.get(pred, "unknown")

# ----------------------------------------------------
# Manual test mode
# ----------------------------------------------------
if __name__ == "__main__":
    while True:
        t = input("Enter: ")
        print("Intent:", predict_intent(t))
