#!/usr/bin/env python3
"""
concat_fasta.py   –
Put reference sequences first, then the rest.

Usage:
    python concat_fasta.py reference.fasta sequences.fasta all_sequence.fasta
"""

import sys
from Bio import SeqIO


def write_fasta_with_reference(ref_path: str, seq_path: str, out_path: str) -> None:
    """
    Write FASTA records so that all references come first,
    followed by the sequences from `seq_path`.

    Every record ends with exactly one newline, so the next
    header ('>') always begins on a fresh line.
    """
    with open(out_path, "w") as out_handle:
        # -- 1. write reference record(s) -------------------------------
        for rec in SeqIO.parse(ref_path, "fasta"):
            # Biopython formats the record; we force ONE trailing '\n'
            out_handle.write(rec.format("fasta").rstrip("\n") + "\n")

        # -- 2. stream the remaining sequences -------------------------
        SeqIO.write(SeqIO.parse(seq_path, "fasta"), out_handle, "fasta")

    print(f"[OK] Wrote '{out_path}' (reference + {seq_path}).")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage: python concat_fasta.py reference.fasta sequences.fasta all_sequence.fasta")

    ref_file, seq_file, out_file = sys.argv[1:4]
    write_fasta_with_reference(ref_file, seq_file, out_file)
