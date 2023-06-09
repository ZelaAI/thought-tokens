# Abstract class for sequences
class Sequence:
    length: int
    
    def __lt__(self, other):
        raise NotImplementedError