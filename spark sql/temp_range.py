from pyspark import SparkConf, SparkContext, SQLContext
import sys, json, re, math
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

#Defining Input and output directories
inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('Weather Data')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#Defining the schema
ischema = StructType([
    StructField('station', StringType(), False),
    StructField('date', StringType(), False),
    StructField('observation', StringType(), False),
    StructField('value', IntegerType(), False),
    StructField('useless', StringType(), False),
    StructField('quality_flag', StringType(), False)])

#Reading the csv file
df = sqlContext.read.format('com.databricks.spark.csv').load(inputs, schema=ischema).cache()

#Filtering TMIN and TMAX values and renaming the columns as min_temp and max_temp
min_temp = df.filter((df.observation == 'TMIN') & (df.quality_flag == '')).withColumnRenamed('value', 'min_temp').select('date', 'station', 'min_temp')
max_temp = df.filter((df.observation == 'TMAX') & (df.quality_flag == '')).withColumnRenamed('value', 'max_temp').select('date', 'station', 'max_temp')
df.unpersist()

#Performing join operation to combine date, station and corresponding min_temp & max_temp values
match_condition = [min_temp.station == max_temp.station, min_temp.date == max_temp.date]
temp_range = min_temp.join(max_temp, match_condition, 'inner').select(min_temp.station, min_temp.date, 'min_temp', 'max_temp')


#Computing range from min_temp and max_temp values
temp_range = temp_range.withColumn('range', temp_range.max_temp - temp_range.min_temp).select('date', 'station', 'range')

#Finding the max range for the specific date and grouping them by date
station_max_range = temp_range.groupby('date').max('range').withColumnRenamed('max(range)', 'max_range')

#Final join operation to select date, station and the max_range, sorting them in ascending order
join_groupby_condition = [station_max_range.date == temp_range.date, station_max_range.max_range == temp_range.range]
day_max_range = station_max_range.join(temp_range, join_groupby_condition, 'inner').select(station_max_range.date, 'station', temp_range.range)
day_max_range = day_max_range.sort(day_max_range.date.asc())

#Converting dataframe to rdd
rdd_convert = day_max_range.rdd

#Mapping the row objects in the right format
rdd_format = rdd_convert.map(lambda b: b['date'] + ' ' + b['station'] + ' ' + str(b['range'])).coalesce(1)

#Saving the mapped data to Text file in the output directory
rdd_format.saveAsTextFile(output)
