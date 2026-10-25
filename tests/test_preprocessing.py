from src.data_preprocessing import preprocessing_stages

def test_pipeline_excludes_label(spark):
    df=spark.createDataFrame([(700.0,20.0,0)],['credit_score','dti','default'])
    stages, nums, cats=preprocessing_stages(df)
    assert 'default' not in nums and 'default' not in cats and stages
