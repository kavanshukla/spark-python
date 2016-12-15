from pyspark import SparkConf, SparkContext
import json
import sys

def score_by_avg((c, avg)):

        return c['score']/avg, c['author']


def add_tuples(a, b):
        return tuple(sum(p) for p in zip(a,b))

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('reddit relative score')
sc = SparkContext(conf=conf)

data=sc.textFile(inputs).map(json.loads)
data.cache()

data_avg=data.map(lambda x: (x['subreddit'], (x['score'], 1)))


avg=data_avg.reduceByKey(add_tuples).mapValues(lambda (u,v): 1.0*u/v)
avg=avg.filter(lambda v: v[1]>0 )


data_comment=data.map(lambda d: (d['subreddit'], d))

c_avg=data_comment.join(avg)

score_and_author=c_avg.values().map(score_by_avg)

score_and_author=score_and_author.sortBy(lambda a: a[0]*-1)
score_and_author.saveAsTextFile(output)
