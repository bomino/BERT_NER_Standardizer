import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the labeled data (50 items)
labeled_data = [
    {"tokens": ["11737", "5/8", "-", "4", "-", "8", "TYPE", "X", "DRYWALL"], "ner_tags": ["B-ITEMNUM", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["42418", "PURDY", "2.5", "-", "IN", "XL", "GLIDE", "ANG"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "I-SPEC"]},
    {"tokens": ["ITEM", "#", "476043", "TAPCON", "1/4", "-", "IN", "X", "7", "-", "IN", "SDS", "DRILLBIT"], "ner_tags": ["O", "O", "B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "B-TYPE"]},
    {"tokens": ["35818CSJ", ";", "3-5/8", "\"", ",", "18GA", ",", "CSJ", "W/", "1-5/8", "\"", "FLANGE", "STUD", ",", "3-5/8", "\"", "X", "10'"], "ner_tags": ["B-ITEMNUM", "O", "B-SIZE", "I-SIZE", "O", "B-SPEC", "O", "B-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "O", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE"]},
    {"tokens": ["11770", "PLUS", "3", "ALL", "PURP", "L/W", "4.5", "-", "G"], "ner_tags": ["B-ITEMNUM", "B-TYPE", "I-TYPE", "B-SPEC", "I-SPEC", "I-SPEC", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY"]},
    {"tokens": ["1147887", "1", "LB", "DECK", "PLUS", "BLACK", "10D", "X", "2"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "B-BRAND", "I-BRAND", "B-SPEC", "B-SIZE", "I-SIZE", "I-SIZE"]},
    {"tokens": ["1/2", "-", "IN", "X", "3", "-", "FT", "X", "5", "-", "FT", "DUR", "EDGE"], "ner_tags": ["B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "B-TYPE", "I-TYPE"]},
    {"tokens": ["10391", "60", "-", "LB", "TYPE", "-", "N", "MORTAR", "MIX"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "I-TYPE"]},
    {"tokens": ["GARAGE", "DRYWALL", "AND", "INSULATION", "MATERIAL"], "ner_tags": ["O", "B-TYPE", "O", "B-TYPE", "O"]},
    {"tokens": ["VALSPAR", "9", "-", "IN", "HEAVY", "DUTY", "T"], "ner_tags": ["B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "O"]},
    {"tokens": ["1984648", "EZ", "ANCOR", "100", "CT", "TWSTLCK", "75"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "I-BRAND", "B-QUANTITY", "I-QUANTITY", "B-TYPE", "B-SPEC"]},
    {"tokens": ["1035775", "PROJECT", "SOURCE", "UTILITY", "BRUSH"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "I-BRAND", "B-SPEC", "B-TYPE"]},
    {"tokens": ["1275111", "10.1", "-", "OZ", "DYNAFLEX", "FAST", "DRY", "LT"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-TYPE", "B-SPEC", "I-SPEC", "I-SPEC"]},
    {"tokens": ["1348401", "GTR", "5", "-", "IN", "8", "-", "H", "H/L", "DISC", "180#", "15", "-", "PC"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "I-SPEC", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY"]},
    {"tokens": ["1/4", "-", "IN", "X", "3", "-", "FT", "X", "5", "-", "FT", "HARDIE", "BACKR", "ITEM#", "11640"], "ner_tags": ["B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "B-BRAND", "B-TYPE", "O", "B-ITEMNUM"]},
    {"tokens": ["45123", "WHIZZ", "4", "-", "IN", "FOAM", "MINI", "ROLLER"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["NELSON", "12", "-", "CT", "WOOD", "SHIMS"], "ner_tags": ["B-BRAND", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "B-TYPE"]},
    {"tokens": ["1175606", "DOUBLE", "MAGNETIC", "CATCH", "BLK"], "ner_tags": ["B-ITEMNUM", "B-SPEC", "I-SPEC", "B-TYPE", "B-SPEC"]},
    {"tokens": ["11751", "USG", "READY", "MIX", "A/P", "4.5", "-", "GAL"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-TYPE", "I-TYPE", "B-SPEC", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY"]},
    {"tokens": ["112265", "PURDY", "9", "-", "IN", "WHITE", "DOVE", "3", "-", "CT"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY"]},
    {"tokens": ["839681", "NTN", "BF", "4-1/2", "-", "IN", "X", "0.045", "T"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "I-SIZE", "O"]},
    {"tokens": ["5249337", "GORILLA", "SG", "MP", "GEL", "5.5", "G", "(-"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "I-SPEC", "B-TYPE", "B-QUANTITY", "I-QUANTITY", "O"]},
    {"tokens": ["955631", "18", "X", "30", "SEWAGE", "BASIN", "LID", "(7"], "ner_tags": ["B-ITEMNUM", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE", "O"]},
    {"tokens": ["GREAT", "STUFF", "12", "-", "FL", "OZ", "GAP/CRK", "FOAM"], "ner_tags": ["B-BRAND", "I-BRAND", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "B-TYPE"]},
    {"tokens": ["53314", "ROMAN", "GH57", "PREMIUM", "WP", "PASTE", "GAL"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "B-QUANTITY"]},
    {"tokens": ["10177", "LNX", "12", "-", "IN", "6TPI", "DEMO", "RECP"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["1054324", "MINWAX", "GRAY", "WOOD", "FILLER"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["1087662", "BSH", "200", "-", "FT", "12V", "3-PLANE", "LASER"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["5480837", "3M", "GENERAL", "PURP", "DUCT", "TAPE"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "I-SPEC", "B-TYPE", "I-TYPE"]},
    {"tokens": ["1064567", "20", "-", "CT", "SWIFFER", "WET", "CLOTHS", "WOOD"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-BRAND", "B-SPEC", "B-TYPE", "I-SPEC"]},
    {"tokens": ["1098062", "PURDY", "2", "-", "IN", "CLEARCUT", "DALE"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC"]},
    {"tokens": ["MAS", "30", "6PNL", "MLD", "TX", "RH", "FLAT", "NK"], "ner_tags": ["B-BRAND", "B-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "I-SPEC"]},
    {"tokens": ["313785", "BERCOM", "HANDY", "PAINT", "TRAY"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "B-TYPE", "I-TYPE"]},
    {"tokens": ["1147862", "1", "LB", "-", "TH", "CR", "DRY", "PH", "6", "X", "1-5/8"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "I-SPEC", "I-SPEC", "B-SIZE", "I-SIZE", "I-SIZE"]},
    {"tokens": ["508899", "VALSPAR", "4", "3/8", "WHITE", "WOVEN"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "B-SPEC", "I-SPEC"]},
    {"tokens": ["40009", "PS", "2.5", "-", "QT", "MEASURE", "CONTAIN"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "B-TYPE"]},
    {"tokens": ["1155564", "DRIVE", "NAIL", "ANCHOR", "1/4", "X", "1"], "ner_tags": ["B-ITEMNUM", "B-SPEC", "I-SPEC", "B-TYPE", "B-SIZE", "I-SIZE", "I-SIZE"]},
    {"tokens": ["11749", "NG", "250", "FT", "PROFORM", "JOINT", "TA"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SIZE", "I-SIZE", "B-SPEC", "B-TYPE", "I-TYPE"]},
    {"tokens": ["11778", "USG", "18", "-", "LB", "L/W", "45", "-", "MIN", "DRY"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["51903", "32", "-", "FL", "OZ", "PURPLE", "CPVC", "AND", "PVC", "PRIMER"], "ner_tags": ["B-ITEMNUM", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-SPEC", "I-SPEC", "O", "I-SPEC", "B-TYPE"]},
    {"tokens": ["121363", "BERCOM", "HANDY", "PAIL", "LINERS"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "B-TYPE", "I-TYPE"]},
    {"tokens": ["305805", "SELLARS", "200", "-", "CT", "RAGS", "BOX"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-QUANTITY", "I-QUANTITY", "I-QUANTITY", "B-TYPE", "B-PACKAGING"]},
    {"tokens": ["5183707", "VALSPAR", "FOAM", "MINI", "ROLLER"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "I-SPEC", "B-TYPE"]},
    {"tokens": ["3834", "BH", "FL", "WASHERS", "SAE", "NO.8", "10"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "B-TYPE", "I-SPEC", "I-SPEC", "B-QUANTITY"]},
    {"tokens": ["78066", "SIKABOND", "CONSTRUCTION", "ADH"], "ner_tags": ["B-ITEMNUM", "B-BRAND", "B-SPEC", "B-TYPE"]},
    {"tokens": ["10", "PC", ",", "10'", "3-5/8", "TRACK"], "ner_tags": ["B-QUANTITY", "I-QUANTITY", "O", "B-SIZE", "I-SIZE", "B-TYPE"]},
    {"tokens": ["114DWS-F5", "FS1145", "/", "VB368S", "1", "1/4", "\"", "FINE", "POINT", "DRYWALL", "SCREWS", "-", "5", "LB"], "ner_tags": ["B-ITEMNUM", "O", "O", "O", "B-SIZE", "I-SIZE", "I-SIZE", "B-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "O", "B-QUANTITY", "I-QUANTITY"]},
    {"tokens": ["260500", "-", "CONCEALOC", "-", "HIDDEN", "FASTENER", "PAIL", "-", "500SF", "-", "FOR", "TIMBERTECH", "/", "AZEK"], "ner_tags": ["B-ITEMNUM", "O", "B-BRAND", "O", "B-SPEC", "B-TYPE", "B-PACKAGING", "O", "B-QUANTITY", "O", "O", "O", "O", "O"]},
    {"tokens": ["35818CSJ", ";", "3-5/8", "\"", ",", "18GA", ",", "CSJ", "W/", "1-5/8", "\"", "FLANGE", "STUD", ",", "3-5/8", "\"", "X", "10'"], "ner_tags": ["B-ITEMNUM", "O", "B-SIZE", "I-SIZE", "O", "B-SPEC", "O", "B-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "I-SPEC", "B-TYPE", "O", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE"]},
    {"tokens": ["1016TSB", ";", "1000T125-54", ";", "10", "\"", "16GA", ",", "STRUCTURAL", "TRACK", "1-1/4", "\"", "LEG", ",", "10", "\"", "X", "10'"], "ner_tags": ["B-ITEMNUM", "O", "O", "O", "B-SIZE", "I-SIZE", "B-SPEC", "O", "I-SPEC", "B-TYPE", "I-SPEC", "I-SPEC", "I-SPEC", "O", "B-SIZE", "I-SIZE", "I-SIZE", "I-SIZE"]}
]

# Define label set
label_list = ["O", "B-BRAND", "I-BRAND", "B-TYPE", "I-TYPE", "B-SIZE", "I-SIZE", 
              "B-SPEC", "I-SPEC", "B-QUANTITY", "I-QUANTITY", "B-PACKAGING", "I-PACKAGING", "B-ITEMNUM", "I-ITEMNUM"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Convert tags to IDs
for item in labeled_data:
    item["ner_tags"] = [label2id[tag] for tag in item["ner_tags"]]

# Split into train and validation sets (80% train, 20% validation)
train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
logger.info(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Load tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=32)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens (CLS, SEP, PAD)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Subword tokens
            previous_word_idx = word_idx
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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert_ner_results",
    eval_strategy="epoch",  # Updated from evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
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