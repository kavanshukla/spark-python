from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, FloatType, ArrayType
import operator

conf = SparkConf().setAppName('Anomaly Detection')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

class AnomalyDetection():

	def readData(self, filename):
		self.rawDF = sqlCt.read.parquet(filename).cache()

	def cat2Num(self, df, indices):

		features = []
		#finding distinct values of 2 columns
		first_feature = df.select(df.rawFeatures[indices[0]]).distinct().collect()
		second_feature = df.select(df.rawFeatures[indices[1]]).distinct().collect()
		
		firstFeature = []
		secondFeature = []
		
		#generating list of distinct values of 2 columns
		for row in first_feature:
			firstFeature.append(row[0])
			
		for row in second_feature:
			secondFeature.append(row[0])
			
		def oneHotEncoder(rawFeatures):
			features = []
			firstColumn = [0.0]*len(firstFeature)
			firstColumn[firstFeature.index(rawFeatures[indices[0]])] = 1.0
			features+= firstColumn
			secondColumn = [0.0]*len(secondFeature)
			secondColumn[secondFeature.index(rawFeatures[indices[1]])] = 1.0
			features += secondColumn
			
			for i in range(2, len(rawFeatures)):
				features += [float(rawFeatures[i])]

			return features
			
		One_Hot_Encode = udf(oneHotEncoder, ArrayType(FloatType(), containsNull=False))

		return df.withColumn('features', One_Hot_Encode(df.rawFeatures))
		

	def addScore(self, df):

		countOfClusters = df.groupBy(df.prediction).count().collect()
		clusterCountList = []
		clusterCountDict = {}
		print countOfClusters
		
		for cluster in countOfClusters:
			clusterCountList.append(float(cluster[1]))
			clusterCountDict[int(cluster[0])] = float(cluster[1])
			
		N_max = max(clusterCountList)
		N_min = min(clusterCountList)
		
		def score(prediction):
			score = (N_max-clusterCountDict[int(prediction)])/(N_max-N_min)
			
			return score
			
		calculate_score = udf(score, DoubleType())	
			
		return df.withColumn('score', calculate_score(df.prediction))
			
	

	def detect(self, k, t):
		#Encoding categorical features using one-hot.
		df1 = self.cat2Num(self.rawDF, [0, 1]).cache()
		df1.show()

		#Clustering points using KMeans
		features = df1.select("features").rdd.map(lambda row: row[0]).cache()
		model = KMeans.train(features, k, maxIterations=40, runs=10, initializationMode="random", seed=20)

		#Adding the prediction column to df1
		modelBC = sc.broadcast(model)
		predictUDF = udf(lambda x: modelBC.value.predict(x), StringType())
		df2 = df1.withColumn("prediction", predictUDF(df1.features)).cache()
		df2.show()

		#Adding the score column to df2; The higher the score, the more likely it is an anomaly 
		df3 = self.addScore(df2).cache()
		df3.show()    

		return df3.where(df3.score > t)
		
		
if __name__ == "__main__":
	ad = AnomalyDetection()
	ad.readData('/user/kavans/logs-features-sample')
	anomalies = ad.detect(8, 0.97)
	print anomalies.count()
	anomalies.show()