from Bio import SeqIO
import sys

fasta = sys.argv[1]  # Input FASTA file
output_fasta = sys.argv[2]  # Output FASTA file

ref_header = "hCoV-19/Wuhan/WIV04/2019|EPI_ISL_402124"

# Read all records
raw_records = list(SeqIO.parse(fasta, "fasta"))
if raw_records[0].description.strip() == ref_header:
    print("Reference genome detected — removing first record.")
    raw_records = raw_records[1:]

# Process records
records = []
for record in raw_records:
    record.id = record.id.split("|")[1]  # Extract ID from description
    
    record.seq = record.seq.upper()  # normalize sequence to uppercase
    record.description = ""  # clear full header
    records.append(record)

# Write updated records
SeqIO.write(records, output_fasta, "fasta")
print(f"Saved {len(records)} cleaned sequences to {output_fasta}")
    
