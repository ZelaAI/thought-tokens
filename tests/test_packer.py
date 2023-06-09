from dataclasses import dataclass
import sys

from data.sequence import Sequence
sys.path.append('../..')
from data.packer import Packer
import pytest

@dataclass
class SequenceTiny(Sequence):
    length: int
    max_new_tokens: int

    def __lt__(self, other):
        return self.max_new_tokens >= other.max_new_tokens


@pytest.fixture
def max_seq_len():
    return 10


@pytest.fixture
def packer(max_seq_len):
    return Packer(max_seq_len)


def test_pack_sequences(packer):
    sequences = [SequenceTiny(10, 10), SequenceTiny(5, 5), SequenceTiny(3, 3), SequenceTiny(3, 3)]
    packer.add_to_queue(sequences)

    # Packer2 just ensures no side effects from packer, somehow this was an issue previously.
    packer2 = Packer(10)
    sequences2 = [SequenceTiny(10, 10), SequenceTiny(5, 5), SequenceTiny(3, 3), SequenceTiny(3, 3)]
    packer2.add_to_queue(sequences2)

    expected_packs = [[10], [5, 3], [3]]
    actual_packs = packer.to_list()
    actual_packs = [[item.length for item in pack] for pack in actual_packs]

    assert expected_packs == actual_packs


def test_sequence_order(packer):
    sequences = [SequenceTiny(10, 5), SequenceTiny(10, 10)]
    packer.add_to_queue(sequences)

    expected_order = [[10], [5]]  # max_new_tokens order
    actual_packs = packer.to_list()
    actual_order = [[item.max_new_tokens for item in pack] for pack in actual_packs]

    assert expected_order == actual_order


def test_dynamic_queue_addition(packer):
    sequences_1 = [SequenceTiny(10, 10), SequenceTiny(5, 5)]
    sequences_2 = [SequenceTiny(3, 3), SequenceTiny(3, 3)]

    packer.add_to_queue(sequences_1)
    first_pack = next(packer)

    packer.add_to_queue(sequences_2)
    second_pack = next(packer)

    expected_first_pack = [10]
    expected_second_pack = [5, 3]

    actual_first_pack = [item.length for item in first_pack]
    actual_second_pack = [item.length for item in second_pack]

    assert expected_first_pack == actual_first_pack
    assert expected_second_pack == actual_second_pack
