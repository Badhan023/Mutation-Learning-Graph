from datetime import date

class Sequence:
    def __init__(self, id, date, seq, mutation):
        self.id = id
        self.date = date
        self.seq = seq
        self.mutation = mutation
        
    def display(self):
        print(f"Genome ID: {self.id}")
        print(f"Collection Date: {self.date}")
        print(f"Sequence: {self.seq}")
        print(f"Mutations: {self.mutation}")
        
        