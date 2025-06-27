from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys
import json  # assuming mutations are in a JSON dict

ref_file = sys.argv[1]  # input reference genome file
directory = sys.argv[2]  # directory of the dataset

mutation_json = f"{directory}/updated_mutation.json"  # input mutation dictionary file
old_mutation_json = f"{directory}/mutation.json"  # with the original dataset mutations
output_fasta = f"{directory}/hypothetical_alignments.fasta"  # output FASTA file


# --- Step 1: Load reference genome and slice coding region (266-29674) ---
ref_record = SeqIO.read(ref_file, "fasta")
ref_seq = ref_record.seq[265:29674]  # 0-based slicing, keep positions 266–29674

# --- Step 2: Load hypothetical mutation dictionary ---

with open(mutation_json) as f:
    mutation_dict = json.load(f)
    
with open(old_mutation_json) as f:
    old_mutation_dict = json.load(f)

# --- Step 3: Build hypothetical genomes ---
hypothetical_records = []

for node_name, mutation_list in mutation_dict.items():
    if node_name not in old_mutation_dict:
        
        # Start with the reference sequence (as a list for mutability)
        genome = list(ref_seq)

        for pos, base in mutation_list:
            zero_based_pos = pos - 266  # Adjust to coding-region-only index
            if 0 <= zero_based_pos < len(genome):
                genome[zero_based_pos] = base.upper()  # Apply mutation

        # Create FASTA record
        mutated_seq = Seq("".join(genome))
        record = SeqRecord(mutated_seq, id=node_name, description="hypothetical variant")
        hypothetical_records.append(record)

# --- Step 4: Save to FASTA ---
with open(output_fasta, "w") as f_out:
    SeqIO.write(hypothetical_records, f_out, "fasta")

print(f"✅ {len(hypothetical_records)} hypothetical sequences saved")