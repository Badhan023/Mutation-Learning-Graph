from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys

def remove_gaps_from_fasta(input_fasta, output_fasta):
    cleaned_records = []

    for record in SeqIO.parse(input_fasta, "fasta"):
        cleaned_seq_str = str(record.seq).replace("-", "")
        cleaned_seq = Seq(cleaned_seq_str)  # ✅ Wrap with Seq()
        cleaned_record = SeqRecord(
            seq=cleaned_seq,
            id=record.id,
            description=record.description
        )
        cleaned_records.append(cleaned_record)

    SeqIO.write(cleaned_records, output_fasta, "fasta")
    print(f"✅ Cleaned FASTA saved to: {output_fasta}")

# Usage from command line
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_gaps.py <input.fasta> <output.fasta>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    remove_gaps_from_fasta(input_file, output_file)
