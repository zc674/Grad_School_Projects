#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as F

# from pyspark.sql import saveMode

def main(spark, train_file, valid_file, test_file):
    '''Main routine for supervised training
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    conf = SparkConf()
    conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '16g'), 
                                            ('spark.driver.memory','16g')])
    spark.sparkContext.stop()
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Load the dataframe
    train = spark.read.parquet(train_file)
    valid = spark.read.parquet(valid_file)
    test = spark.read.parquet(test_file)

    # Give the dataframe a temporary view so we can run SQL queries
    train.createOrReplaceTempView('train')
    valid.createOrReplaceTempView('valid')
    test.createOrReplaceTempView('test')

    print("load data")

    # Subsample train set
    df = train.withColumn("rowID", F.monotonically_increasing_id())
    df = df.repartition("user_id")
    df.createOrReplaceTempView("df")
    sample = spark.sql("SELECT * FROM df ORDER BY rowID DESC LIMIT 5000000")
    sample = sample.repartition("user_id")
    sample.createOrReplaceTempView("sample")

    # Create index
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx").fit(sample)
    item_indexer = StringIndexer(inputCol="track_id", outputCol="item_idx", handleInvalid="skip").fit(sample)
    print("fit successully")

    # Transform dataset
    idx_train = user_indexer.transform(sample)
    idx_train = item_indexer.transform(idx_train)

    print("train index")

    idx_valid = user_indexer.transform(valid)
    idx_valid = item_indexer.transform(idx_valid)

    print("valid index")

    idx_test = user_indexer.transform(test)
    idx_test = item_indexer.transform(idx_test)

    print("test index")

    # Create repartition
    idx_train = idx_train.repartition("user_idx")
    idx_valid = idx_valid.repartition("user_idx")
    idx_test = idx_test.repartition("user_idx")

    print("repartion finish")

    # Save indexed dataset
    idx_train.write.format("parquet").mode("overwrite").save("./indexed_train.parquet")
    idx_valid.write.format("parquet").mode("overwrite").save("./indexed_valid.parquet")
    idx_test.write.format("parquet").mode("overwrite").save("./indexed_test.parquet")

    print("save succesfully")



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('sample').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    valid_file = sys.argv[2]
    test_file = sys.argv[3]

    # Call our main routine
    main(spark, train_file, valid_file, test_file)
