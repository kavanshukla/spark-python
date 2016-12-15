from pyspark import SparkConf, SparkContext, SQLContext
import sys, json, re, math
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

#Defining Input and output directories
inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('Weather Data SQL')
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

#Registering table all_weather_data
sqlContext.registerDataFrameAsTable(df, "all_weather_data")

#Filtering TMAX and TMAX values and renaming the columns as min_temp and max_temp
min_temp = sqlContext.sql("""
            SELECT date, station, value as min_temp
            FROM all_weather_data
            WHERE observation="TMIN" AND quality_flag=""
            """)
sqlContext.registerDataFrameAsTable(min_temp, "min_temp")

max_temp = sqlContext.sql("""
            SELECT date, station, value as max_temp
            FROM all_weather_data
            WHERE observation="TMAX" AND quality_flag=""
            """)
sqlContext.registerDataFrameAsTable(max_temp, "max_temp")

#Performing join operation to combine date, station and corresponding min_temp & max_temp values
temp_range = sqlContext.sql("""
                SELECT min_temp.station, min_temp.date, min_temp, max_temp 
                FROM min_temp JOIN max_temp 
                ON (min_temp.station = max_temp.station AND min_temp.date = max_temp.date)       
                """)

sqlContext.registerDataFrameAsTable(temp_range, "temp_range")

#Computing range from min_temp and max_temp values and storing it in temp_range1 table
temp_range1 = sqlContext.sql("""
                SELECT date, station, (max_temp - min_temp) as range
                FROM temp_range
                """)
sqlContext.registerDataFrameAsTable(temp_range1, "temp_range1")

#Finding the max range for the specific date and grouping them by date
station_max_range = sqlContext.sql("""
                    SELECT date, MAX(range) as max_range
                    FROM temp_range1
                    GROUP BY date
                    """)
sqlContext.registerDataFrameAsTable(station_max_range, "station_max_range")

#Final join operation to retreive date, station and the max_range, sorting them in ascending order
day_max_range = sqlContext.sql("""
                    SELECT station_max_range.date, temp_range1.station, temp_range1.range 
                    FROM station_max_range JOIN temp_range1 
                    ON (station_max_range.date = temp_range1.date AND station_max_range.max_range = temp_range1.range)
                    ORDER BY date ASC
                    """)

#Converting dataframe to rdd
rdd_convert = day_max_range.rdd

#Mapping the row objects in the right format
rdd_format = rdd_convert.map(lambda b: b['date'] + ' ' + b['station'] + ' ' + str(b['range'])).coalesce(1)

#Saving the mapped data to Text file in the output directory
rdd_format.saveAsTextFile(output)
