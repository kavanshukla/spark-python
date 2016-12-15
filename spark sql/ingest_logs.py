from pyspark import SparkConf, SparkContext, SQLContext
import sys, re, math
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import datetime as dt
from pyspark.sql import *

def main():
	conf = SparkConf().setAppName('ingest logs')
	sc = SparkContext(conf=conf)
	sqlContext = SQLContext(sc)

	inputs = sys.argv[1] 
	output = sys.argv[2]

	#Reading the input file, and then matching the pattern
	file_data = sc.textFile(inputs)
	linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
	
	#Mapping the data after fetching the required values out of the Nasa Web server logs file
	KeyValue = file_data.map(lambda line : linere.split(line)).filter(lambda x : len(x)==6).map(lambda y : (y[1],(dt.datetime.strptime(y[2], '%d/%b/%Y:%H:%M:%S')),y[3],y[-2])).cache()
	
	#Mapping the KeyValue RDD as the required format of 4 columns
	Nasa = KeyValue.map(lambda p: {"host": p[0], "datetime": p[1], "path": p[2], "bytes": long(p[3])})
	
	#Converting Nasa to DataFrame and then registering it as Table
	schemaNasa = sqlContext.createDataFrame(Nasa)
	schemaNasa.registerTempTable("NasaLogs")

	#Writing the data into a parquet file
	schemaNasa.write.format('parquet').save(output)
	
	#Reading the data from Parquet file and then Registering it in Table Format
	parquetdata = sqlContext.read.parquet(output)
	parquetdata.registerTempTable("parquetTable")

	#Firing SQL query to count the total number of bytes transferred using SUM(bytes)
	totalbytes = sqlContext.sql("""
    	SELECT SUM(bytes)
    	FROM parquetTable
	""")
	totalbytes.show()

if __name__ == "__main__":
	main()