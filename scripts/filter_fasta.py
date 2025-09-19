#!/usr/bin/env python3
from Bio import SeqIO
import argparse

def filter_fasta(input_fasta, output_fasta):
    """
    Remove sequences containing 'N' from a FASTA file.
    Print counts before and after filtering.
    """
    # Read all records
    records = list(SeqIO.parse(input_fasta, "fasta"))
    total_before = len(records)

    # Filter out sequences with N
    filtered_records = [record for record in records if "N" not in str(record.seq)]
    total_after = len(filtered_records)

    # Write filtered records
    with open(output_fasta, "w") as outfile:
        SeqIO.write(filtered_records, outfile, "fasta")

    # Print summary
    print(f"Total sequences before filtering: {total_before}")
    print(f"Total sequences after filtering:  {total_after}")
    print(f"Sequences removed: {total_before - total_after}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter out FASTA sequences containing N.")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("output_fasta", help="Output FASTA file (filtered)")
    args = parser.parse_args()

    filter_fasta(args.input_fasta, args.output_fasta)
