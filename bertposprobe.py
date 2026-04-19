import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from transformers import (
    BertConfig, 
    BertForTokenClassification, 
    AutoTokenizer, 
    DataCollatorForTokenClassification
)

# ==========================================
# 0. Setup & Reproducibility
# ==========================================
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Dataset & Preprocessing (POS Specific)
# ==========================================
print("Loading dataset and tokenizer...")
dataset = load_dataset("lhoestq/conll2003")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
NUM_POS_LABELS = 47

def tokenize_and_align_pos_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128
    )
    labels = []
    for i, label in enumerate(examples["pos_tags"]):
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

print("Preprocessing dataset for POS...")
pos_dataset = dataset.map(tokenize_and_align_pos_labels, batched=True)
columns_to_keep = ["input_ids", "attention_mask", "labels"]
pos_dataset.set_format(type="torch", columns=columns_to_keep)

pos_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
pos_train_loader = DataLoader(pos_dataset["train"], batch_size=16, shuffle=True, collate_fn=pos_collator)
pos_test_loader = DataLoader(pos_dataset["test"], batch_size=16, collate_fn=pos_collator)

# ==========================================
# 2. Training & Eval Helpers
# ==========================================
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    loop = tqdm(loader, leave=True, desc="Training Probe")
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
# 3. Shared Model Configuration
# ==========================================
# Using the custom dimensions from your training backbone
pos_config = BertConfig(
    vocab_size=30522, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128,
    num_labels=NUM_POS_LABELS
)

# ==========================================
# 4. PROBE 1: Linear Probing on POS (SSL / MLM)
# ==========================================
print("\n" + "="*50)
print("--- PROBE 1: Evaluating SSL (MLM) Representations ---")
print("="*50)

if not os.path.exists("best_mlm_model.pth"):
    print("ERROR: 'best_mlm_model.pth' not found. Ensure it is in the same directory.")
else:
    pos_ssl_model = BertForTokenClassification(pos_config).to(device)
    print("Loading MLM weights into POS Classification model...")
    # strict=False is required here because the saved MLM model has a different head
    pos_ssl_model.load_state_dict(torch.load("best_mlm_model.pth"), strict=False)

    print("Freezing BERT base...")
    for name, param in pos_ssl_model.named_parameters():
        if "classifier" not in name: 
            param.requires_grad = False

    optimizer_pos_ssl = AdamW(pos_ssl_model.classifier.parameters(), lr=1e-3) 

    best_test_loss = float("inf")
    patience = 5
    epochs_no_improve = 0
    max_epochs = 30

    for epoch in range(max_epochs): 
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss = train_epoch(pos_ssl_model, pos_train_loader, optimizer_pos_ssl, device)
        test_loss, test_accuracy = evaluate(pos_ssl_model, pos_test_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | POS Acc: {test_accuracy:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Saving with a distinct name so it doesn't overwrite your base models
            torch.save(pos_ssl_model.state_dict(), "best_pos_ssl_probe_model.pth")
            epochs_no_improve = 0
            print(" -> SSL POS Probe improved and saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" -> No improvement for {patience} epochs. Early stopping!")
                break

# ==========================================
# 5. PROBE 2: Linear Probing on POS (SL / Supervised)
# ==========================================
print("\n" + "="*50)
print("--- PROBE 2: Evaluating SL (Supervised) Representations ---")
print("="*50)

if not os.path.exists("best_pos_model.pth"):
    print("ERROR: 'best_pos_model.pth' not found. Ensure it is in the same directory.")
else:
    sl_model = BertForTokenClassification(pos_config).to(device)
    print("Loading Supervised POS weights into model...")
    # strict=True (default) works here because best_pos_model.pth was already a TokenClassification model
    sl_model.load_state_dict(torch.load("best_pos_model.pth"))
    
    print("Freezing BERT base and resetting classifier head...")
    for param in sl_model.bert.parameters():
        param.requires_grad = False
    
    # Completely replace the head to reset weights and train them from scratch
    sl_model.classifier = torch.nn.Linear(pos_config.hidden_size, NUM_POS_LABELS).to(device)
    
    optimizer_pos_sl = AdamW(sl_model.classifier.parameters(), lr=1e-3) 
    
    best_test_loss = float("inf")
    patience = 5
    epochs_no_improve = 0
    max_epochs = 30
    
    for epoch in range(max_epochs): 
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        train_loss = train_epoch(sl_model, pos_train_loader, optimizer_pos_sl, device)
        test_loss, test_accuracy = evaluate(sl_model, pos_test_loader, device) 
        
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | POS Acc: {test_accuracy:.4f}")
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            # Saving with a distinct name so it doesn't overwrite your base models
            torch.save(sl_model.state_dict(), "best_pos_sl_probe_model.pth")
            epochs_no_improve = 0
            print(" -> SL POS Probe improved and saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f" -> No improvement for {patience} epochs. Early stopping!")
                break

print("\n🎉 Probing complete!")