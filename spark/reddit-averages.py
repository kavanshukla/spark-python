from pyspark import SparkConf, SparkContext
import sys, json
 
inputs = sys.argv[1]
output = sys.argv[2]

def add_pair((a1,b1),(a2,b2)):
    return (a1+a2,b1+b2)
    
    
conf = SparkConf().setAppName('Reddit Average')
sc = SparkContext(conf=conf)

file_data = sc.textFile(inputs)

json_data = file_data.map(json.loads)

json_extract = json_data.map(lambda line: (line['subreddit'], (float(line['score']),1)))
wordcount = json_extract.reduceByKey(add_pair).coalesce(1).cache()
average = wordcount.map(lambda (key,(total,count)): (key,float(total/count)))

json_average = average.map(lambda x:json.dumps(x))
json_average.saveAsTextFile(output);
