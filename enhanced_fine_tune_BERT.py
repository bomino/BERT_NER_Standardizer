import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import json
import csv
import os
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to load labeled data from a file
def load_labeled_data(file_path: str):
    """Load labeled data from JSON or CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file '{file_path}' not found in the directory.")
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)
    elif file_path.endswith('.csv'):
        labeled_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            current_item = None
            for row in reader:
                if row['tokens']:  # New item starts with non-empty tokens
                    if current_item:
                        labeled_data.append(current_item)
                    current_item = {"tokens": [], "ner_tags": []}
                if current_item:
                    current_item["tokens"].append(row['tokens'])
                    current_item["ner_tags"].append(row['ner_tags'])
            if current_item:  # Append the last item
                labeled_data.append(current_item)
    else:
        raise ValueError("Unsupported file format. Use .json or .csv.")
    
    logger.info(f"Loaded {len(labeled_data)} labeled items from {file_path}")
    return labeled_data

# Define label set
label_list = ["O", "B-BRAND", "I-BRAND", "B-TYPE", "I-TYPE", "B-SIZE", "I-SIZE", 
              "B-SPEC", "I-SPEC", "B-QUANTITY", "I-QUANTITY", "B-PACKAGING", "I-PACKAGING", "B-ITEMNUM", "I-ITEMNUM"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Load data from file
data_file = "output.json"  # Change to "labeled_data.csv" if using CSV
labeled_data = load_labeled_data(data_file)

# Convert tags to IDs
for item in labeled_data:
    item["ner_tags"] = [label2id[tag] for tag in item["ner_tags"]]

# Split into train and validation sets (80% train, 20% validation)
train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
logger.info(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")

# Check label distribution
all_labels = [label for item in train_data for label in item["ner_tags"]]
label_counts = Counter(all_labels)
logger.info("Label distribution in training data:")
for label, count in label_counts.items():
    logger.info(f"{id2label[label]}: {count}")

# Oversample rare labels if necessary
rare_labels = [label for label, count in label_counts.items() if count < 10]
if rare_labels:
    rare_examples = [item for item in train_data if any(label in rare_labels for label in item["ner_tags"])]
    train_data.extend(rare_examples)
    logger.info(f"Oversampled {len(rare_examples)} examples with rare labels.")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize and align labels (enhanced to label all subwords)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=32)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens (CLS, SEP, PAD)
            else:
                label_ids.append(label[word_idx])  # Assign label to all subwords
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

# Load BERT model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", 
                                                    num_labels=len(label_list),
                                                    id2label=id2label,
                                                    label2id=label2id)

# Define training arguments with adjusted learning rate and epochs
training_args = TrainingArguments(
    output_dir="./bert_ner_results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # Adjusted learning rate
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=15,  # Increased epochs
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Define a simple compute_metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[l for l in label if l != -100] for label in labels]
    pred_labels = [[p for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    flat_true = [item for sublist in true_labels for item in sublist]
    flat_pred = [item for sublist in pred_labels for item in sublist]
    accuracy = accuracy_score(flat_true, flat_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(flat_true, flat_pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
logger.info("Starting training...")
trainer.train()

# Save the fine-tuned model
model.save_pretrained("bert_construction_ner")
tokenizer.save_pretrained("bert_construction_ner")
logger.info("Model saved to 'bert_construction_ner'")

# Test the model on a sample
from transformers import pipeline
ner_pipeline = pipeline("ner", model="bert_construction_ner", tokenizer="bert_construction_ner", aggregation_strategy="simple")
test_description = "ITEM # 476043 TAPCON 1/4-IN X 7-IN SDS DRILLBIT"
result = ner_pipeline(test_description)
print("Test Prediction:")
for entity in result:
    print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Start: {entity['start']}, End: {entity['end']}")