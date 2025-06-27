import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from Bio import SeqIO
from tqdm import tqdm
import sys

# Optionally disable FlashAttention bags (model itself has no flash attention code)
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
os.environ["DISABLE_FLASH_ATTN"] = "1"

# Use the no‐flashattention DNABERT‑2 variant
MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"

# Load configuration, tokenizer, and model
config = BertConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
)

# Ensure model returns dict (if supported)
model.config.return_dict = True

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Parse CLI args
FASTA_FILE = sys.argv[1]    # Input FASTA file
OUTPUT_FILE = sys.argv[2]   # Output .npy file for embeddings
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

    # Tokenize and move to device
    inputs = tokenizer(
        batch_seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward
    with torch.no_grad():
        outputs = model(**inputs)
        # if tuple, first element is last_hidden_state
        hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        # CLS token embedding ([batch, hidden_dim])
        cls_emb = hidden[:, 0, :].cpu().numpy()

    # Collect
    all_embs.extend(cls_emb)

# Save
np.save(OUTPUT_FILE, np.array(all_embs))
print(f"Embeddings saved to {OUTPUT_FILE}")
