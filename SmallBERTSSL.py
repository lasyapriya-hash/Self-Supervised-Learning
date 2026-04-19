import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    BertForTokenClassification, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification
)
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

# ==========================================
# 1. Load Dataset & Tokenizer
# ==========================================
dataset = load_dataset("lhoestq/conll2003")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
NUM_NER_LABELS = 9

# ==========================================
# 2. Preprocessing for Stage 1: MLM
# ==========================================
def tokenize_mlm(examples):
    # Just tokenize the words, the collator will handle creating the masked labels
    return tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

mlm_dataset = dataset.map(tokenize_mlm, batched=True)
# Remove original text columns so only input_ids and attention_mask remain
mlm_dataset = mlm_dataset.remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"])
mlm_dataset.set_format(type="torch")

# 15% Masking Collator
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

mlm_train_loader = DataLoader(mlm_dataset["train"], batch_size=16, shuffle=True, collate_fn=mlm_collator)
mlm_val_loader = DataLoader(mlm_dataset["validation"], batch_size=16, collate_fn=mlm_collator)

# ==========================================
# 3. Preprocessing for Stage 2: NER
# ==========================================
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
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
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

ner_dataset = dataset.map(tokenize_and_align_labels, batched=True)
columns_to_keep = ["input_ids", "attention_mask", "labels"]
ner_dataset.set_format(type="torch", columns=columns_to_keep)

ner_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
ner_train_loader = DataLoader(ner_dataset["train"], batch_size=16, shuffle=True, collate_fn=ner_collator)
ner_test_loader = DataLoader(ner_dataset["test"], batch_size=16, collate_fn=ner_collator)

# ==========================================
# 4. Standard Training & Eval Loops
# ==========================================
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

def evaluate(model, loader, device, task="mlm"):
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

# Shared Base Configuration
base_config = BertConfig(
    vocab_size=30522, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128
)

# ==========================================
# STAGE 1: Self-Supervised Learning (MLM)
# ==========================================
print("\n--- STAGE 1: Pre-training via Masked Language Modeling (SSL) ---")
mlm_model = BertForMaskedLM(base_config).to(device)
optimizer_mlm = AdamW(mlm_model.parameters(), lr=5e-5) # Slightly higher LR for MLM

best_val_loss = float("inf")
patience_mlm = 10  # Higher patience because MLM is noisy and slow
epochs_no_improve_mlm = 0
max_epochs_mlm = 150 # Give it plenty of runway

for epoch in range(max_epochs_mlm): 
    train_loss = train_epoch(mlm_model, mlm_train_loader, optimizer_mlm, device)
    val_loss, val_accuracy = evaluate(mlm_model, mlm_val_loader, device, task="mlm")
    
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MLM Acc: {val_accuracy:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(mlm_model.state_dict(), "best_mlm_model.pth")
        epochs_no_improve_mlm = 0
        print(" -> MLM model improved and saved!")
    else:
        epochs_no_improve_mlm += 1
        print(f" -> No improvement for {epochs_no_improve_mlm} epoch(s).")
        if epochs_no_improve_mlm >= patience_mlm:
            print("\nEarly stopping triggered for MLM pre-training!")
            break

# ==========================================
# STAGE 2: Linear Probing on NER
# ==========================================
print("\n--- STAGE 2: Linear Probing on NER ---")
# 1. Update config for Token Classification
ner_config = base_config
ner_config.num_labels = NUM_NER_LABELS

# 2. Initialize a fresh Token Classification model
ner_model = BertForTokenClassification(ner_config).to(device)

# 3. Load the pretrained BERT base weights (strict=False is the magic trick here)
print("Loading MLM weights into Token Classification model...")
ner_model.load_state_dict(torch.load("best_mlm_model.pth"), strict=False)

# 4. Freeze the BERT encoder (Linear Probing)
for name, param in ner_model.named_parameters():
    if "classifier" not in name: # Lock everything except the new classifier head
        param.requires_grad = False

# 5. Train ONLY the classifier head
optimizer_ner = AdamW(ner_model.classifier.parameters(), lr=1e-3) # Higher LR for linear probing

best_test_loss = float("inf")
patience_ner = 5
epochs_no_improve_ner = 0
max_epochs_ner = 30

for epoch in range(max_epochs_ner): 
    train_loss = train_epoch(ner_model, ner_train_loader, optimizer_ner, device)
    test_loss, test_accuracy = evaluate(ner_model, ner_test_loader, device, task="ner")
    
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | NER Acc: {test_accuracy:.4f}")
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(ner_model.state_dict(), "best_ner_ssl_probe_model.pth")
        epochs_no_improve_ner = 0
        print(" -> SSL Probe improved and saved!")
    else:
        epochs_no_improve_ner += 1
        print(f" -> No improvement for {epochs_no_improve_ner} epoch(s).")
        if epochs_no_improve_ner >= patience_ner:
            print("\nEarly stopping triggered for NER linear probe!")
            break

# Load the absolute best NER model for final confirmation
ner_model.load_state_dict(torch.load("best_ner_ssl_probe_model.pth"))
print(f"\nSSL Training Complete! Peak NER Test Loss: {best_test_loss:.4f}")