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

conf = SparkConf().setAppName('Movie Recommendation-item_item')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

train_data = sc.textFile("MovieLens100K_train.txt")
test_data = sc.textFile("MovieLens100K_test.txt")

ratingsRDD = train_data.map(lambda x: x.split('\t')).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
ratings = sqlContext.createDataFrame(ratingsRDD).cache()
ratings.show()
test_ratingsRDD = test_data.map(lambda x: x.split('\t')).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
test_ratings = sqlContext.createDataFrame(test_ratingsRDD).cache()
test_ratings.registerTempTable('test_ratings')
ratings.registerTempTable('ratings')
user_average_rating = sqlContext.sql("""
        SELECT userId, AVG(rating) as avg_rating
        FROM ratings
        GROUP BY userId
        ORDER BY userId
        """)
user_average_rating.show()

user_average_rating.registerTempTable('user_average_rating')

cross_join = sqlContext.sql("""
        SELECT user_average_rating.userId, movieId, rating, avg_rating
        FROM ratings CROSS JOIN user_average_rating
        """)
cross_join.registerTempTable('cross_join')
cross_join.show()

stdev = sqlContext.sql("""
    SELECT t.userId , t.movieId, t.rating, t.avg_rating, t.rating-t.avg_rating AS deviation
    FROM cross_join as t
    """)
stdev.show()
stdev.cache()
stdev.registerTempTable('stdev')

user_item_combination = sqlContext.sql("""
    SELECT s1.userId as userId1, s1.movieId as movieId1, s1.rating as rating1, s2.userId as userId2, s2.movieId as movieId2, s2.rating as rating2
    FROM ratings s1, ratings s2
    WHERE s1.movieId != s2.movieId AND s1.userId=s2.userId
    """)
user_item_combination.registerTempTable('user_item_combination')
user_item_combination.cache()
user_item_combination.repartition(200)
user_item_combination.show()

correlation = sqlContext.sql("""
    SELECT movieId1, movieId2, corr(rating1, rating2) as correlation
    FROM user_item_combination
    GROUP BY movieId1, movieId2
    """)
correlation.cache()
correlation.show()
threshold_list = [0.5, 0.6, 0.7, 0.8, 0.9]
rmse_list = []

for i in range(0,5):

	filtered_correlation = correlation.filter(correlation.correlation >= threshold_list[i])
	filtered_correlation.show()
	filtered_correlation.count()

	filtered_correlation.registerTempTable('filtered_correlation')

	joined_correlation = sqlContext.sql("""
    	SELECT *
    	FROM test_ratings JOIN filtered_correlation
    	ON test_ratings.movieId = filtered_correlation.movieId1
    	""")


	joined_correlation = sqlContext.sql("""
    	SELECT userId, movieId, movieId2,rating, correlation
    	FROM test_ratings JOIN filtered_correlation
    	ON test_ratings.movieId = filtered_correlation.movieId1
    	""")
	joined_correlation.show()
	joined_correlation.cache()
	joined_correlation.registerTempTable('joined_correlation')
	train_joined_correlation = sqlContext.sql("""
    	SELECT stdev.userId, joined_correlation.movieId, movieId2, joined_correlation.rating, correlation, deviation , avg_rating
    	FROM stdev JOIN joined_correlation
    	ON stdev.userId = joined_correlation.userId AND stdev.movieId = joined_correlation.movieId2
    	ORDER BY userId
    	""")
	train_joined_correlation.cache()
	train_joined_correlation.registerTempTable('train_joined_correlation')
	train_joined_correlation.repartition(200)
	train_joined_correlation.show()

	formula = sqlContext.sql("""
    	SELECT userId, movieId, rating, SUM(deviation*correlation)/SUM(correlation) as prediction_val, avg_rating
    	FROM train_joined_correlation
    	GROUP BY userId, movieId, avg_rating, rating
    	""")
	formula.repartition(200)
	formula.cache()
	formula.show()
	formula.registerTempTable('formula')

	formula_result = sqlContext.sql("""
    	SELECT userId, movieId,rating, prediction_val + avg_rating as prediction
    	FROM formula
    	""") 
	formula_result.show()

	evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
    	                            predictionCol="prediction")
	rmse = evaluator.evaluate(formula_result)
	print ("Root-mean-square error "  + " = " + str(rmse))
	rmse_list.append("Root-mean-square error for different Similarity cut-off Threshold " + str(threshold_list[i]) + " = " + str(rmse))
print rmse_list