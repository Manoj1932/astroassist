import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# --------------------------
# Load Dataset
# --------------------------
df = pd.read_csv("data/astro_intent_dataset.csv")

texts = df["text"].tolist()
labels = df["intent"].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Save label map
import json
with open("models/label_maps.json", "w") as f:
    json.dump({i: label for i, label in enumerate(label_encoder.classes_)}, f)

# --------------------------
# Tokenizer
# --------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

encodings = tokenizer(texts, truncation=True, padding=True)

# --------------------------
# Torch Dataset
# --------------------------
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

dataset = IntentDataset(encodings, labels_encoded)

# --------------------------
# CLASS WEIGHTS (THIS IS THE FIX)
# --------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_encoded),
    y=labels_encoded
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print("Class Weights:", class_weights_tensor)

# --------------------------
# Model
# --------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

# Override loss function to weighted loss
def compute_weighted_loss(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(logits.device))
    loss = loss_fct(logits, labels)

    return loss

# --------------------------
# Training Arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="models/intent_classifier",
    overwrite_output_dir=True,
    num_train_epochs=12,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_strategy="epoch",
)

# --------------------------
# Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_loss_func=compute_weighted_loss  # << NEW: CUSTOM WEIGHTED LOSS
)

# Train
trainer.train()

print("Training complete! Model saved.")
