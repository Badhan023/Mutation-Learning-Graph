class SerialIDGenerator:
    def __init__(self):
        self.counter = 0
        
    def generate(self):
        self.counter += 1
        id = "t"+str(self.counter)
        return id
    