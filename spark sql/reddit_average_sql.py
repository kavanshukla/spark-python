from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import json
import sys

def main():
	conf = SparkConf().setAppName('reddit relative sql score')
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	inputs = sys.argv[1]
	output = sys.argv[2]
	
	#Defining the Schema
	schema = StructType([
    		StructField('subreddit', StringType(), False),
		StructField('score', IntegerType(), False),
	])

	#Reading the reddit json files
	comments = sqlContext.read.json(inputs, schema)
	
	#Registering the Json data in Table Format and Querying the table to get subreddit names and average scores
	comments.registerTempTable('comments')
	averages = sqlContext.sql("""
   	SELECT subreddit, AVG(score)
    	FROM comments
    	GROUP BY subreddit
	""")

	averages.write.save(output, format='json', mode='overwrite')


if __name__ == "__main__":
	main()
