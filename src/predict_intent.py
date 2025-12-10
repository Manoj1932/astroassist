from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import json
import os

# =========================
# ✅ Load label map FIRST
# =========================
LABEL_MAP_PATH = os.path.join("data", "label_maps.json")

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}
num_labels = len(label_map)

# =========================
# ✅ Use base DistilBERT safely
# =========================
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

model.eval()

# =========================
# ✅ Intent Prediction Logic
# =========================
def predict_intent(text):
    t = text.lower()

    # ✅ HARD EMERGENCY OVERRIDE
    emergency_keywords = [
        "fire", "smoke", "leak", "toxic", "fumes",
        "explosion", "blast", "suffocating", "can't breathe",
        "pressure dropping", "hull breach", "gas leak",
        "support failing", "danger", "emergency"
    ]

    for word in emergency_keywords:
        if word in t:
            return "emergency"

    # ✅ ML Prediction
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        out = model(**enc)
        pred = torch.argmax(out.logits, dim=1).item()

    return id_to_label.get(pred, "unknown")


# =========================
# ✅ Local CLI Test Mode
# =========================
if __name__ == "__main__":
    print("\n✅ Intent Predictor Ready — type a command:")
    while True:
        cmd = input("\nEnter command: ")
        print("Predicted Intent:", predict_intent(cmd))
