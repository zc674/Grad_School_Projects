#!/usr/bin/env python
# -*- coding: utf-8 -*-


# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import functions as F


def main(spark, train_file, test_file, rank, reg, alpha):
    '''Main routine for supervised training
    Parameters
    ----------
    spark : SparkSession object
    data_file : string, path to the parquet file to load
    model_file : string, path to store the serialized model file
    '''

    # Load the dataframe
    train = spark.read.parquet(train_file)
    test = spark.read.parquet(test_file)

    # Give the dataframe a temporary view so we can run SQL queries
    train.createOrReplaceTempView('train')
    test.createOrReplaceTempView('test')

    # Transform count to log compression
    train_log = train.withColumn("log_count", F.log("count"))
    test_log = test.withColumn("log_count", F.log("count"))

    train_log.createOrReplaceTempView('train_log')
    test_log.createOrReplaceTempView('test_log')

    # Build model for input parameters
    rank = float(rank)
    reg = float(reg)
    alpha = float(alpha)

    als = ALS(implicitPrefs=True, userCol="user_idx", itemCol="item_idx", ratingCol="log_count")\
        .setParams(rank=rank, regParam=reg, alpha=alpha) 
    model = als.fit(train_log)

    print("model fitted")
    
    # Create predition and truth lists
    k = 500

    recommendations = model.recommendForUserSubset(test,k)
    perUserRecom = recommendations.selectExpr("user_idx", "recommendations.item_idx as prediction")
    label_list = test.orderBy(F.col("user_idx"), F.expr("count DESC")).groupby("user_idx").agg(F.expr("collect_list(item_idx) as label"))
    perUserItem = label_list.select("user_idx", "label")

    print("predition and label")

    predictionAndLabel = perUserItem.join(perUserRecom, "user_idx").rdd.map(lambda row: (row.prediction, row.label))

    print("inner join")

    # Use ranking metrics for evaluation
    metrics=RankingMetrics(predictionAndLabel)
    mean_precision = metrics.meanAveragePrecision
    k_precision = metrics.precisionAt(k)

    print("At rank={0}, regParam={1}, alpha = {2}, mean average precision is {3}".format(rank, reg, alpha, mean_precision))
    print("At rank={0}, regParam={1}, alpha = {2}, precision at top 500 words is {3}".format(rank, reg, alpha, k_precision))

    pass



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('als_log').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # get parameters
    rank = sys.argv[3]
    reg = sys.argv[4]
    alpha = sys.argv[5]

    # Call our main routine
    main(spark, train_file, test_file, rank, reg, alpha)
    
