# src/predict_intent.py
import os
import json
import onnxruntime as ort
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Try a fallback: if models live at project root `models/`
if not os.path.exists(MODEL_DIR):
    alt = os.path.join(os.path.dirname(BASE_DIR), "models")
    if os.path.exists(alt):
        MODEL_DIR = alt

label_map_path = os.path.join(MODEL_DIR, "label_maps.json")
onnx_model_path = os.path.join(MODEL_DIR, "intent_classifier.onnx")

print("Using MODEL_DIR:", MODEL_DIR)
print("label_maps.json exists?", os.path.exists(label_map_path))
print("intent_classifier.onnx exists?", os.path.exists(onnx_model_path))
print("label_map_path:", label_map_path)
print("onnx_model_path:", onnx_model_path)

if not os.path.exists(label_map_path):
    raise FileNotFoundError(f"label_maps.json not found at {label_map_path}")
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

with open(label_map_path, "r") as f:
    label_map = json.load(f)
id_to_label = {v: k for k, v in label_map.items()}

emergency_keywords = [
    "fire", "smoke", "leak", "toxic", "fumes",
    "explosion", "blast", "suffocating", "can't breathe",
    "pressure dropping", "hull breach", "gas leak",
    "support failing", "danger", "emergency"
]

sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = False
sess_options.enable_cpu_mem_arena = False

session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"], sess_options=sess_options)

# Print the ONNX input names & shapes to know exactly what to feed
print("ONNX model inputs:")
for inp in session.get_inputs():
    print(" ", inp.name, inp.shape, inp.type)

# stable simple hash (deterministic)
def stable_hash(word: str) -> int:
    return abs(int(np.sum([ord(c) * (i + 1) for i, c in enumerate(word)]))) & 0x7fffffff

def preprocess(text: str, max_len: int = 20):
    text = text.lower().strip()
    tokens = text.split()
    ids = [stable_hash(t) % 30000 for t in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    # attention mask: 1 where token > 0 else 0 (assumes padding ID is 0)
    attn = [1 if i != 0 else 0 for i in ids]
    return np.array([ids], dtype=np.int64), np.array([attn], dtype=np.int64)

def predict_intent(text: str):
    t = text.lower()
    # emergency rule
    for word in emergency_keywords:
        if word in t:
            return "emergency"

    input_ids, attention_mask = preprocess(text)

    # Build input dictionary matching ONNX model inputs
    onnx_inputs = {}
    names = [inp.name for inp in session.get_inputs()]

    # Common input names: "input_ids", "attention_mask", maybe "token_type_ids"
    if "input_ids" in names:
        onnx_inputs["input_ids"] = input_ids
    elif names:
        # if model expects a single unnamed input, assign to first
        onnx_inputs[names[0]] = input_ids

    if "attention_mask" in names:
        onnx_inputs["attention_mask"] = attention_mask
    elif "attentionmask" in names:
        onnx_inputs["attentionmask"] = attention_mask

    # If model expects token_type_ids, send zeros
    if "token_type_ids" in names:
        onnx_inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

    # final sanity check: print missing inputs
    required = set([inp.name for inp in session.get_inputs()])
    provided = set(onnx_inputs.keys())
    missing = required - provided
    if missing:
        raise ValueError(f"Required ONNX inputs ({missing}) are missing from prepared feed ({provided})")

    outputs = session.run(None, onnx_inputs)
    pred = int(np.argmax(outputs[0], axis=1)[0])
    return id_to_label.get(pred, "unknown")

if __name__ == "__main__":
    while True:
        t = input("Enter: ")
        print("Intent:", predict_intent(t))
