import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertConfig, 
    BertForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling
)

# ==========================================
# 0. Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists("best_mlm_model.pth"):
    raise FileNotFoundError("ERROR: 'best_mlm_model.pth' not found in the current directory.")

# ==========================================
# 1. Dataset & Preprocessing (MLM Specific)
# ==========================================
print("Loading dataset and tokenizer...")
dataset = load_dataset("lhoestq/conll2003")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_mlm(examples):
    # We only need to tokenize; the collator handles the actual masking
    return tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

print("Preprocessing test dataset for MLM...")
# We only map the test set to save time
test_dataset = dataset["test"].map(tokenize_mlm, batched=True)
test_dataset = test_dataset.remove_columns(["id", "tokens", "pos_tags", "chunk_tags", "ner_tags"])
test_dataset.set_format(type="torch")

# 15% Masking Collator
mlm_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

mlm_test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=mlm_collator)

# ==========================================
# 2. Shared Base Configuration & Model Loading
# ==========================================
base_config = BertConfig(
    vocab_size=30522, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128
)

print("Loading trained MLM model...")
mlm_model = BertForMaskedLM(base_config).to(device)
mlm_model.load_state_dict(torch.load("best_mlm_model.pth"))

# ==========================================
# 3. Evaluation Function
# ==========================================
def evaluate_mlm(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating MLM on Test Set"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            
            # Calculate accuracy only on the MASKED tokens (where labels are not -100)
            preds = torch.argmax(outputs.logits, dim=-1)
            valid_mask = labels != -100
            
            correct += (preds[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()
            
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

# ==========================================
# 4. Execution
# ==========================================
print("\n" + "="*50)
print("--- RUNNING MLM TEST EVALUATION ---")
print("="*50)

test_loss, test_accuracy = evaluate_mlm(mlm_model, mlm_test_loader, device)

print(f"\nFinal Test Results:")
print(f"  MLM Test Loss:     {test_loss:.4f}")
print(f"  MLM Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print("="*50)