from pyspark import SparkConf, SparkContext
from pyspark.sql import Row
from pyspark.sql.functions import lit
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import sys
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import *
from pyspark.mllib.linalg import Vectors

conf = SparkConf().setAppName('Movie Recommendation-ALS')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

output = sys.argv[1]

train_data = sc.textFile("MovieLens100K_train.txt")
test_data = sc.textFile("MovieLens100K_test.txt")

ratingsRDD = train_data.map(lambda x: x.split('\t')).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
ratings = sqlContext.createDataFrame(ratingsRDD).cache()
ratings.show()
test_ratingsRDD = test_data.map(lambda x: x.split('\t')).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
test_ratings = sqlContext.createDataFrame(test_ratingsRDD).cache()
    
rank_list = [2, 4, 8, 16 , 32, 64, 128, 256]
rmse_list = []
for i in range(0,8):
    rankl = rank_list[i]
    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=20, rank= rank_list[i], regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
    
    paramGrid = ParamGridBuilder() \
    .addGrid(als.regParam, [0.1]) \
    .build()
    
    crossval = CrossValidator(estimator=als,
                              estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction"),
                          numFolds=5)  # use 3+ folds in practice
    
    cvModel = crossval.fit(ratings)

    # Evaluate the model by computing the RMSE on the test data
    predictions = cvModel.transform(test_ratings)
    predictions = predictions.filter(predictions.prediction != float('nan'))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error for rank " + str(rank_list[i]) + " = " + str(rmse))
    rmse_list.append("Root-mean-square error for rank " + str(rank_list[i]) + " = " + str(rmse))
    predictions.show()

    movie_factors = cvModel.bestModel.itemFactors
    print movie_factors
    movie_factors.show()


    movie_factors.registerTempTable('movie_factors')

    midDF =  sqlContext.sql("""
        SELECT id, features
        FROM movie_factors
        """)

    midRDD = midDF.rdd
    #midRDD.collect()
    vectorRDD = midRDD.map(lambda (x,y): Row(id = x, features = Vectors.dense(y))).cache()
    vectorRDD.collect()
    kmeans_input = sqlContext.createDataFrame(vectorRDD).cache()
    kmeans = KMeans(featuresCol="features", predictionCol="prediction").setK(50)
    kmeans_df = kmeans.fit(kmeans_input)

    kmeans_transformed = kmeans_df.transform(kmeans_input)
    kmeans_transformed.show()

    kmeans_transformed.registerTempTable('kmeans_table')

    movie_items = sc.textFile("u.item")
    movienameRDD = movie_items.map(lambda x: x.split('|')).map(lambda p: Row(movieId=int(p[0]), movieName=p[1]))
    movienamesDF = sqlContext.createDataFrame(movienameRDD).cache()
    print "movienames_table"
    movienamesDF.show()
    movienamesDF.registerTempTable('movienames_table')


    movie_clusters =  sqlContext.sql("""
        SELECT id as movieId
        FROM kmeans_table
        WHERE prediction IN (20, 30)
        """)
    print "movie_clusters"
    movie_clusters.show()
    movie_clusters.registerTempTable('movie_clusters')

    printmovienames = sqlContext.sql("""
        SELECT movienames_table.movieName
        FROM movienames_table, movie_clusters
        WHERE movienames_table.movieId = movie_clusters.movieId
        """)
    
    final_movie_cluster = printmovienames.rdd

    final_movie_cluster.coalesce(1).saveAsTextFile(output + '/rank' + str(rankl))

print rmse_list