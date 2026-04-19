import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import (
    BertConfig, 
    BertForTokenClassification, 
    BertForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForTokenClassification
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NUM_POS_LABELS = 47
NUM_NER_LABELS = 9
MAX_BATCHES = 30 # Limit the data so t-SNE doesn't take hours (30 batches * 16 = 480 sentences)

# CoNLL-2003 NER Tag Mapping
NER_TAG_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# --- 1. Load Data & Tokenizer ---
print("Loading dataset and tokenizer...")
dataset = load_dataset("lhoestq/conll2003")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

test_dataset = dataset["test"].map(tokenize_and_align_labels, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collator)

# --- 2. Load Models ---
print("Loading saved models...")
base_config = BertConfig(
    vocab_size=30522, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128
)

# Load POS Model (SL)
pos_config = base_config
pos_config.num_labels = NUM_POS_LABELS
pos_model = BertForTokenClassification(pos_config)
pos_model.load_state_dict(torch.load("best_pos_model.pth", map_location=DEVICE))
pos_model.to(DEVICE)
pos_model.eval()

# Load MLM Model (SSL)
mlm_model = BertForMaskedLM(base_config)
mlm_model.load_state_dict(torch.load("best_mlm_model.pth", map_location=DEVICE))
mlm_model.to(DEVICE)
mlm_model.eval()

# --- 3. Feature Extraction Function ---
def extract_embeddings(model, loader, device, max_batches):
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Extracting representations")):
            if i >= max_batches: break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Extract the hidden states from the base BERT encoder
            # .bert gives us the encoder, avoiding the task-specific classification heads
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            
            # Get the last hidden state (the final representations)
            hidden_states = outputs.last_hidden_state # Shape: [batch, seq_len, hidden_size]
            
            # Flatten everything
            hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            labels = labels.view(-1)
            
            # Filter out padding (-100) and "O" tags (0) to focus strictly on named entities
            valid_mask = (labels != -100) & (labels != 0)
            
            valid_embeddings = hidden_states[valid_mask]
            valid_labels = labels[valid_mask]
            
            all_embeddings.append(valid_embeddings.cpu().numpy())
            all_labels.append(valid_labels.cpu().numpy())
            
    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)

# Extract for both models
print("\nProcessing POS (SL) representations...")
pos_embeddings, labels_array = extract_embeddings(pos_model, test_loader, DEVICE, MAX_BATCHES)

print("\nProcessing MLM (SSL) representations...")
mlm_embeddings, _ = extract_embeddings(mlm_model, test_loader, DEVICE, MAX_BATCHES) # Labels are identical

# --- 4. Metric Calculation ---
# Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
# Range is [-1, 1]. Higher is better.
print("\nCalculating Silhouette Scores...")
pos_silhouette = silhouette_score(pos_embeddings, labels_array)
mlm_silhouette = silhouette_score(mlm_embeddings, labels_array)

print(f"-> Supervised (POS) Silhouette Score: {pos_silhouette:.4f}")
print(f"-> Self-Supervised (MLM) Silhouette Score: {mlm_silhouette:.4f}")

# --- 5. t-SNE Dimensionality Reduction ---
print("\nRunning t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)

# Fit t-SNE
print("Fitting POS data...")
pos_tsne = tsne.fit_transform(pos_embeddings)
print("Fitting MLM data...")
mlm_tsne = tsne.fit_transform(mlm_embeddings)

# --- 6. Plotting ---
print("Generating plot...")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Create string labels for the legend
string_labels = [NER_TAG_NAMES[label_idx] for label_idx in labels_array]
unique_labels = sorted(list(set(string_labels)))

# Plot 1: POS Model
sns.scatterplot(
    ax=axes[0], x=pos_tsne[:, 0], y=pos_tsne[:, 1], 
    hue=string_labels, hue_order=unique_labels,
    palette="tab10", s=20, alpha=0.8
)
axes[0].set_title(f"Supervised (POS) Representations\nSilhouette Score: {pos_silhouette:.3f}")
axes[0].legend(loc="best", fontsize=9)

# Plot 2: MLM Model
sns.scatterplot(
    ax=axes[1], x=mlm_tsne[:, 0], y=mlm_tsne[:, 1], 
    hue=string_labels, hue_order=unique_labels,
    palette="tab10", s=20, alpha=0.8
)
axes[1].set_title(f"Self-Supervised (MLM) Representations\nSilhouette Score: {mlm_silhouette:.3f}")
axes[1].legend(loc="best", fontsize=9)

plt.suptitle("t-SNE Visualization of Hidden States (Named Entities Only)", fontsize=16)
plt.tight_layout()
plt.savefig("representation_comparison.png", dpi=300)
print("\nSuccess! Saved plot as 'representation_comparison.png'")