"""Dataset class to be extended by dataset-specific classes."""
from pathlib import Path

"""Dataset class to be extended by dataset-specific classes."""

class Dataset:
    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / 'data'


