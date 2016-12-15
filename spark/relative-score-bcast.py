from pyspark import SparkConf, SparkContext
import json
import sys

def join_comment_avg(com, avg_dic):
	key = com[0]
	avg = avg_dic.value[key]
	return score_by_avg((com[1], avg))

def score_by_avg((c, avg)):

        return c['score']/avg, c['author']


def add_tuples(a, b):
        return tuple(sum(p) for p in zip(a,b))

inputs = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('reddit relative score bcast')
sc = SparkContext(conf=conf)

data=sc.textFile(inputs).map(json.loads)
data.cache()

data_avg=data.map(lambda x: (x['subreddit'], (x['score'], 1)))


avg=data_avg.reduceByKey(add_tuples).mapValues(lambda (u,v): 1.0*u/v)
avg=avg.filter(lambda v: v[1]>0 )


data_comment=data.map(lambda d: (d['subreddit'], d))

avg_dic = sc.broadcast(dict(avg.collect()))


score_and_author=avg.values().map(lambda (sr,score):score/avg_dic.value['subreddit'])

score_and_author=data_comment.map(lambda com: join_comment_avg(com, avg_dic))

score_and_author=score_and_author.sortBy(lambda a: a[0]*-1)
score_and_author.saveAsTextFile(output)
