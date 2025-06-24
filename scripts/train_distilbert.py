

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import json

# Label mapping
label2id = {
    "O": 0,
    "B-Product": 1,
    "I-Product": 2,
    "B-PRICE": 3,
    "I-PRICE": 4,
    "B-LOC": 5,
    "I-LOC": 6
}

# Parse CoNLL data
def parse_conll(file_path):
    sentences = []
    tokens, tags = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                if len(line.split()) == 2:
                    token, tag = line.split()
                    tokens.append(token)
                    tags.append(label2id.get(tag, 0))
    return sentences

# Load dataset
# Make sure 'data/clean/labeled_conll.txt' exists and is correctly formatted.
# You might need to create these directories and files if they don't exist.
try:
    dataset = parse_conll("data/clean/labeled_conll.txt")
except FileNotFoundError:
    print("Error: 'data/clean/labeled_conll.txt' not found. Please ensure the file exists and the path is correct.")
    # Create dummy data for demonstration if the file is not found
    print("Creating dummy data for demonstration purposes...")
    dataset = [
        {"tokens": ["This", "is", "a", "sample", "Product", "at", "100", "Birr", "in", "Addis"], 
         "ner_tags": [0, 0, 0, 0, 1, 0, 3, 4, 0, 5]},
        {"tokens": ["Another", "item", "for", "250", "Dollars", "from", "Gonder"], 
         "ner_tags": [0, 0, 0, 3, 4, 0, 5]}
    ]
    # Optionally, write this dummy data to the expected file path
    import os
    os.makedirs("data/clean", exist_ok=True)
    with open("data/clean/labeled_conll.txt", "w", encoding="utf-8") as f:
        for sent in dataset:
            for token, tag_id in zip(sent["tokens"], sent["ner_tags"]):
                tag_name = [k for k, v in label2id.items() if v == tag_id][0] # Reverse lookup
                f.write(f"{token} {tag_name}\n")
            f.write("\n")


hf_dataset = Dataset.from_list(dataset)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    # Remove 'offset_mapping' as it's not needed for the model input
    tokenized_inputs.pop("offset_mapping")
    return tokenized_inputs

tokenized_dataset = hf_dataset.map(tokenize_and_align_labels, batched=True)

# Split dataset
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test['train']
eval_dataset = train_test['test']

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels=len(label2id),
    id2label={v: k for k, v in label2id.items()},
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./outputs/ner-distilbert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",  # Corrected argument name
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=1,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train
print("Starting training...")
trainer.train()
print("Training complete.")

# Evaluate and save metrics
print("Evaluating model...")
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# Ensure the output directory exists
import os
os.makedirs("outputs", exist_ok=True)
with open("outputs/distilbert_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Metrics saved to outputs/distilbert_metrics.json")

# Save model
os.makedirs("models/ner-distilbert", exist_ok=True)
trainer.save_model("models/ner-distilbert")
print("Model saved to models/ner-distilbert")