import os
import sys

# Standard overrides for Mac safety
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from Bio import SeqIO
from tqdm import tqdm

# SWITCH: Use the official stable research repository
MODEL_NAME = "zhihan1996/DNABERT-2-117M"

print(f"Loading official model config and tokenizer from {MODEL_NAME}...")
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Explicitly default back to standard CPU mechanics
device = torch.device("cpu")
print("Loading model weights directly into CPU memory...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
)

model.config.return_dict = True
model.to(device).eval()
print("Model loaded successfully!")

# Parse CLI args
FASTA_FILE = sys.argv[1]     # Input FASTA file
OUTPUT_FILE = sys.argv[2]    # Output .npy file for embeddings
BATCH_SIZE = int(sys.argv[3])  # Batch size

# Read sequences
sequences = {r.id: str(r.seq) for r in SeqIO.parse(FASTA_FILE, "fasta")}
ids = list(sequences.keys())
seqs = list(sequences.values())
all_embs = []

print(f"Processing {len(seqs)} sequences in batches of {BATCH_SIZE}...")

# Batch loop
for idx in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Extracting Embeddings"):
    batch_ids = ids[idx: idx + BATCH_SIZE]
    batch_seqs = seqs[idx: idx + BATCH_SIZE]

    inputs = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        cls_emb = hidden[:, 0, :].cpu().numpy()

    all_embs.extend(cls_emb)

np.save(OUTPUT_FILE, np.array(all_embs))
print(f"Embeddings saved to {OUTPUT_FILE}")