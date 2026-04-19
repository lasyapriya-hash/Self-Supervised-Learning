import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import BertConfig, BertForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from tqdm import tqdm
import random
import numpy as np

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 1. Load Dataset
dataset = load_dataset("lhoestq/conll2003")

# CoNLL-2003 has 47 POS tags and 9 NER tags
NUM_POS_LABELS = 47
NUM_NER_LABELS = 9

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Preprocessing Function (Crucial for Token Classification)
def tokenize_and_align_labels(examples, label_column):
    # CoNLL-2003 is already split into words, so we must tell the tokenizer
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples[label_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens (CLS, SEP, PAD) get -100 so PyTorch ignores them in loss
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first sub-token of a given word
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # If a word is split into multiple sub-tokens by BERT, ignore the rest
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Process for Task A (POS)
pos_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, "pos_tags"), batched=True)
# Process for Task B (NER)
ner_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, "ner_tags"), batched=True)

# Format datasets
columns_to_keep = ["input_ids", "attention_mask", "labels"]
pos_dataset.set_format(type="torch", columns=columns_to_keep)
ner_dataset.set_format(type="torch", columns=columns_to_keep)

# Data Collator for Token Classification (pads dynamically if we weren't already max_length)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Loaders for Task A (POS)
pos_train_loader = DataLoader(pos_dataset["train"], batch_size=16, shuffle=True, collate_fn=data_collator)
pos_val_loader = DataLoader(pos_dataset["validation"], batch_size=16, collate_fn=data_collator)

# Loaders for Task B (NER)
ner_train_loader = DataLoader(ner_dataset["train"], batch_size=16, shuffle=True, collate_fn=data_collator)
ner_test_loader = DataLoader(ner_dataset["test"], batch_size=16, collate_fn=data_collator)

# 3. Model Configuration (Notice we use TokenClassification now, not MaskedLM)
config = BertConfig(
    vocab_size=30522,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=128,
    num_labels=NUM_POS_LABELS # Start with Task A labels
)

model = BertForTokenClassification(config)
model.to(device)

# Standard training loops (kept mostly the same as yours)
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True, desc="Training")
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            valid_mask = labels != -100
            
            correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
    return total_loss / len(loader), (correct / total if total > 0 else 0.0)

# ==========================================
# STAGE 1: Train Full Model on Task A (POS)
# ==========================================
print("\n--- STAGE 1: Full Fine-Tuning on POS Tags ---")
optimizer_pos = AdamW(model.parameters(), lr=3e-5)

best_val_loss = float("inf")
patience_pos = 5
epochs_no_improve_pos = 0
max_epochs_pos = 50

for epoch in range(max_epochs_pos): 
    train_loss = train_epoch(model, pos_train_loader, optimizer_pos, device)
    val_loss, val_accuracy = evaluate(model, pos_val_loader, device)
    
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    
    # Check for improvement
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_pos_model.pth")
        epochs_no_improve_pos = 0
        print(" -> Model improved and saved!")
    else:
        epochs_no_improve_pos += 1
        print(f" -> No improvement for {epochs_no_improve_pos} epoch(s).")
        if epochs_no_improve_pos >= patience_pos:
            print("\nEarly stopping triggered for POS training!")
            break

# Load the best POS model before moving to Stage 2
model.load_state_dict(torch.load("best_pos_model.pth"))
print("\nLoaded best POS model for Stage 2.")

# ==========================================
# STAGE 2: Linear Probing on Task B (NER)
# ==========================================
print("\n--- STAGE 2: Linear Probing on NER Tags ---")

# 1. Freeze the BERT base model
for param in model.bert.parameters():
    param.requires_grad = False

# 2. Replace the classification head for NER (this automatically has requires_grad=True)
model.classifier = torch.nn.Linear(config.hidden_size, NUM_NER_LABELS)
model.num_labels = NUM_NER_LABELS
model.to(device)

# 3. Create a NEW optimizer that ONLY updates the classifier head
optimizer_ner = AdamW(model.classifier.parameters(), lr=1e-3) 

best_test_loss = float("inf")
patience_ner = 5
epochs_no_improve_ner = 0
max_epochs_ner = 30

for epoch in range(max_epochs_ner): 
    train_loss = train_epoch(model, ner_train_loader, optimizer_ner, device)
    # Using the test loader here since that's what was in your original code
    test_loss, test_accuracy = evaluate(model, ner_test_loader, device) 
    
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "best_ner_probe_model.pth")
        epochs_no_improve_ner = 0
        print(" -> Probe improved and saved!")
    else:
        epochs_no_improve_ner += 1
        print(f" -> No improvement for {epochs_no_improve_ner} epoch(s).")
        if epochs_no_improve_ner >= patience_ner:
            print("\nEarly stopping triggered for NER linear probe!")
            break

# Load the absolute best NER model for final confirmation
model.load_state_dict(torch.load("best_ner_probe_model.pth"))
print(f"\nTraining Complete! Peak NER Test Loss: {best_test_loss:.4f}")