import sys, re, string, os, gzip
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
import cassandra
import datetime as dt
from pyspark import SparkConf
import pyspark_cassandra


def main():
	#Defining input directory, keyspace and table name
	inputs = sys.argv[1]
	keyspace = sys.argv[2]

	#Cluster configuration
	cluster_seeds = ['199.60.17.136', '199.60.17.173']
	conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds))
	sc = pyspark_cassandra.CassandraSparkContext(conf=conf)

	#Reading the input file, and then matching the pattern
	file_data = sc.textFile(inputs)
	linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
	
	#Mapping the data after fetching the required values out of the Nasa Web server logs file
	KeyValue = file_data.map(lambda line : linere.split(line)).filter(lambda x : len(x)==6).map(lambda y : (y[1],(dt.datetime.strptime(y[2], '%d/%b/%Y:%H:%M:%S')),y[3],y[-2])).cache()
	
	#Mapping the KeyValue RDD as the required format of 4 columns
	Nasa = KeyValue.map(lambda p: {"host": p[0], "datetime": p[1], "path": p[2], "bytes": long(p[3])})
	
	Nasa.saveToCassandra(keyspace, 'nasalogs')

if __name__ == "__main__":
	main()
