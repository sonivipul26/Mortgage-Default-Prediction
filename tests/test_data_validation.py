import pytest
from src.data_validation import validate_observations

def test_rejects_missing_target(spark):
    with pytest.raises(ValueError): validate_observations(spark.createDataFrame([('a',)],['loan_sequence_number']))

def test_accepts_binary_target(spark):
    assert validate_observations(spark.createDataFrame([('a',0),('b',1)],['loan_sequence_number','default']))
