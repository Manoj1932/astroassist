from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import json
import os

intent_model_path = os.path.join("models", "intent_classifier")

tokenizer = DistilBertTokenizerFast.from_pretrained(intent_model_path)
model = DistilBertForSequenceClassification.from_pretrained(intent_model_path)

with open("data/label_maps.json") as f:
    label_map = json.load(f)

id_to_label = {v: k for k, v in label_map.items()}

def predict_intent(text):
    t = text.lower()

    # ========== ✅ HARD EMERGENCY RULE ==========
    emergency_keywords = [
        "fire", "smoke", "leak", "toxic", "fumes",
        "explosion", "blast", "suffocating", "can't breathe",
        "pressure dropping", "hull breach", "gas leak",
        "support failing", "danger", "emergency"
    ]

    for word in emergency_keywords:
        if word in t:
            return "emergency"
        
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**enc)
        pred = torch.argmax(out.logits, dim=1).item()
    return id_to_label[pred]

if __name__ == "__main__":
    print("\n Intent Predictor Ready — type a command:")
    while True:
        cmd = input("\nEnter command: ")
        print("Predicted Intent:", predict_intent(cmd))
