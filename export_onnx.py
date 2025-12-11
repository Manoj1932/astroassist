from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

model_name = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

model.eval()

dummy_input = tokenizer("hello world", return_tensors="pt")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "intent_classifier.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "logits": {0: "batch"},
    },
    opset_version=13
)

print("ONNX model exported!")
