import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from transformers import BertConfig, BertForTokenClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

# ==========================================
# 1. Setup & Data Loading
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
NUM_POS_LABELS = 47

# Load dataset and setup loaders (Reuse your previous preprocessing logic)
dataset = load_dataset("lhoestq/conll2003")

def tokenize_and_align_pos_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
    labels = []
    for i, label in enumerate(examples["pos_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None: label_ids.append(-100)
            elif word_idx != previous_word_idx: label_ids.append(label[word_idx])
            else: label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

pos_dataset = dataset.map(tokenize_and_align_pos_labels, batched=True)
pos_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_loader = DataLoader(pos_dataset["test"], batch_size=16, collate_fn=DataCollatorForTokenClassification(tokenizer))

# Custom config from your backbone
pos_config = BertConfig(
    vocab_size=30522, hidden_size=256, num_hidden_layers=4,
    num_attention_heads=4, intermediate_size=1024, max_position_embeddings=128,
    num_labels=NUM_POS_LABELS, output_hidden_states=True # Crucial for similarity
)

# ==========================================
# 2. Extraction Function
# ==========================================
def collect_embeddings(model, loader):
    model.eval()
    embeddings_by_class = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get hidden states from the backbone
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state # [batch, seq_len, hidden_dim]
            
            # Filter valid tokens (not padding/sub-words)
            mask = labels != -100
            for i in range(labels.shape[0]):
                valid_indices = mask[i].nonzero(as_tuple=True)[0]
                for idx in valid_indices:
                    label_id = labels[i, idx].item()
                    vec = last_hidden_state[i, idx]
                    embeddings_by_class[label_id].append(vec)
                    
    # Convert lists to tensors
    centroids = {}
    for label_id, vecs in embeddings_by_class.items():
        stacked_vecs = torch.stack(vecs)
        embeddings_by_class[label_id] = stacked_vecs
        centroids[label_id] = torch.mean(stacked_vecs, dim=0)
        
    return embeddings_by_class, centroids

# ==========================================
# 3. Similarity Logic
# ==========================================
def compute_metrics(embeddings_by_class, centroids):
    # 1. Intra-class similarity (Avg similarity of members to their class centroid)
    intra_sims = []
    for label_id, vecs in embeddings_by_class.items():
        centroid = centroids[label_id].unsqueeze(0)
        sim = F.cosine_similarity(vecs, centroid)
        intra_sims.append(sim.mean().item())
    
    avg_intra = np.mean(intra_sims)
    
    # 2. Inter-class similarity (Avg similarity between different centroids)
    inter_sims = []
    centroid_list = list(centroids.values())
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            sim = F.cosine_similarity(centroid_list[i].unsqueeze(0), centroid_list[j].unsqueeze(0))
            inter_sims.append(sim.item())
            
    avg_inter = np.mean(inter_sims)
    
    return avg_intra, avg_inter

# ==========================================
# 4. Execution
# ==========================================
def run_analysis(model_path, name, is_mlm=False):
    print(f"\n--- Analyzing {name} ---")
    model = BertForTokenClassification(pos_config).to(device)
    # Load weights (MLM requires strict=False because of the head mismatch)
    model.load_state_dict(torch.load(model_path), strict=(not is_mlm))
    
    embs, centers = collect_embeddings(model, test_loader)
    intra, inter = compute_metrics(embs, centers)
    
    print(f"Results for {name}:")
    print(f"  Avg Intra-Class Similarity (Clustering): {intra:.4f}")
    print(f"  Avg Inter-Class Similarity (Separation): {inter:.4f}")
    print(f"  Separation Gap (Intra - Inter): {intra - inter:.4f}")
    return intra, inter

# Run for both
ssl_intra, ssl_inter = run_analysis("best_mlm_model.pth", "SSL (MLM) Model", is_mlm=True)
sl_intra, sl_inter = run_analysis("best_pos_model.pth", "SL (Supervised) Model", is_mlm=False)