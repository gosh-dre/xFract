import pytest
from components import data_ingestion

def test_load_config():
    config = data_ingestion.load_config()
    assert isinstance(config, dict)