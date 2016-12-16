from pyspark import SparkConf, SparkContext
import sys
import operator
import re, unicodedata, string
import pyspark_cassandra
import math


def rdd_for(keyspace, table, sc, split_size=None):
        rdd = sc.cassandraTable(keyspace, table, split_size=split_size,
        row_format=pyspark_cassandra.RowFormat.DICT).setName(table)
        return rdd

def add_tuples(a, b):
        return tuple(sum(p) for p in zip(a,b))

def get_tuples(line):
        return(line['host'], (int(line['bytes']), 1))

def calculate(v, bytes_avg, requests_avg):
        c1 = (v[1][0] - bytes_avg) * (v[1][1] - requests_avg)
        c2 = (v[1][0] - bytes_avg)**2
        c3 = (v[1][1] - requests_avg) **2

        return (c1,c2,c3)

def main():
        keyspace = sys.argv[1]
        output = sys.argv[2]

        cluster_seeds = ['199.60.17.136', '199.60.17.173']
        conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds))
        conf.setAppName('correlate logs better in cassandra')

        sc = pyspark_cassandra.CassandraSparkContext(conf=conf)


        mapped_lines = rdd_for(keyspace, "nasalogs", sc, split_size=9000)

        mapped_tuples = mapped_lines.map(get_tuples)
        reduced_tuples = mapped_tuples.reduceByKey(add_tuples)

        reduced_tuples.cache()

        N = reduced_tuples.count()

        average_compute = reduced_tuples.map(lambda p: (p[1][0], p[1][1]) )
        bytes_total,requests_total = average_compute.reduce(add_tuples)

        bytes_avg = 1.0*bytes_total/N
        requests_avg = 1.0*requests_total/N


        temp = reduced_tuples.map(lambda p: calculate(p, bytes_avg, requests_avg))
        numerator,denom_1,denom_2 = temp.reduce(add_tuples)
        denom_1 = math.sqrt(denom_1)
        denom_2 = math.sqrt(denom_2)
        denom = denom_1*denom_2
        result = numerator/denom

        temp = ['r = %f' % (result), 'r^2 = %f' % (result**2)]
        sc.parallelize(temp, numSlices=1).saveAsTextFile(output)

if __name__ == "__main__":
    main()
