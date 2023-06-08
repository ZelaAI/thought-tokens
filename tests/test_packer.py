import sys
sys.path.append('../..')
import unittest
from data.packer import Packer
import unittest
from data.sequence import Sequence

class Sequence:
    def __init__(self, length, max_new_tokens):
        self.length = length
        self.max_new_tokens = max_new_tokens
    
    def __lt__(self, other):
        return self.max_new_tokens >= other.max_new_tokens

class TestPacker(unittest.TestCase):
    def setUp(self):
        self.max_seq_len = 10

    def test_pack_sequences(self):
        packer = Packer(self.max_seq_len)
        sequences = [Sequence(10, 10), Sequence(5, 5), Sequence(3, 3), Sequence(3, 3)]
        packer.add_to_queue(sequences)
        
        # Packer2 just ensures no side effects from packer, somehow this was an issue previously.
        packer2 = Packer(self.max_seq_len)
        sequences2 = [Sequence(10, 10), Sequence(5, 5), Sequence(3, 3), Sequence(3, 3)]
        packer2.add_to_queue(sequences2)

        expected_packs = [[10], [5, 3], [3]]
        actual_packs = packer.to_list()
        actual_packs = [[item.length for item in pack] for pack in actual_packs]

        self.assertEqual(expected_packs, actual_packs)

    def test_sequence_order(self):
        packer = Packer(self.max_seq_len)
        sequences = [Sequence(10, 5), Sequence(10, 10)]
        packer.add_to_queue(sequences)

        expected_order = [[10], [5]]  # max_new_tokens order
        actual_packs = packer.to_list()
        actual_order = [[item.max_new_tokens for item in pack] for pack in actual_packs]

        self.assertEqual(expected_order, actual_order)

    def test_dynamic_queue_addition(self):
        packer = Packer(self.max_seq_len)
        sequences_1 = [Sequence(10, 10), Sequence(5, 5)]
        sequences_2 = [Sequence(3, 3), Sequence(3, 3)]
        
        packer.add_to_queue(sequences_1)
        first_pack = next(packer)

        packer.add_to_queue(sequences_2)
        second_pack = next(packer)

        expected_first_pack = [10]
        expected_second_pack = [5, 3]
        
        actual_first_pack = [item.length for item in first_pack]
        actual_second_pack = [item.length for item in second_pack]

        self.assertEqual(expected_first_pack, actual_first_pack)
        self.assertEqual(expected_second_pack, actual_second_pack)


if __name__ == '__main__':
    unittest.main()