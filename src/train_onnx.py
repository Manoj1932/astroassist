import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import onnx
from transformers.convert_graph_to_onnx import convert

# ====== LOAD CSV ======
df = pd.read_csv("data/astro_intent_dataset.csv")

texts = df["text"].tolist()
labels = df["label"].tolist()

# ====== ENCODE LABELS ======
encoder = LabelEncoder()
labels_enc = encoder.fit_transform(labels)

label_map = {label: int(idx) for idx, label in enumerate(encoder.classes_)}
id_to_label = {int(v): k for k, v in label_map.items()}

import json
with open("data/label_maps.json", "w") as f:
    json.dump(label_map, f, indent=4)

# ====== DATASET CLASS ======
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ====== LOAD MODEL ======
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_map)
)

dataset = IntentDataset(texts, labels_enc, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# ====== TRAIN ======
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)

print("Training...")

for epoch in range(1):
    for batch in loader:
        optim.zero_grad()
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        output.loss.backward()
        optim.step()

print("Training done!")

# ====== SAVE PYTORCH MODEL ======
torch.save(model.state_dict(), "onnx_model/pytorch_model.bin")
tokenizer.save_pretrained("onnx_model")

# ====== EXPORT TO ONNX ======
print("Exporting to ONNX...")

convert(
    framework="pt",
    model=model,
    tokenizer=tokenizer,
    opset=11,
    output="onnx_model/intent_model.onnx",
    pipeline_name="text-classification"
)

print("ONNX model exported successfully!")
