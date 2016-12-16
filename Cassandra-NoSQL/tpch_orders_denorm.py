from pyspark import SparkConf, SparkContext, SQLContext
import sys
import operator
import re, unicodedata, string
import pyspark_cassandra


def rdd_for_orders(keyspace, sc, sqlContext, orderkeys):

        table="orders_parts"
        create_table = sc.cassandraTable(keyspace, table, split_size=10000).select('orderkey', 'totalprice', 'part_names')
        create_table = create_table.where("orderkey in (" + ",".join(orderkeys) + ")")
        create_table = create_table.setName(table)
        return create_table

def map_output(value):
        return 'Order #%s $%.2f: %s' % (value['orderkey'], value['totalprice'], ', '.join(value['part_names']))

def main():
        keyspace = sys.argv[1]
        output = sys.argv[2]
        orderkeys = sys.argv[3:]

        cluster_seeds = ['199.60.17.136', '199.60.17.173']
        conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds))
        conf.set('spark.dynamicAllocation.maxExecutors', 20)
        conf.setAppName('tpch orders denorm cassandra')
        sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
        sqlContext = SQLContext(sc)

        orders_rdd = rdd_for_orders(keyspace, sc, sqlContext, orderkeys)

        orders_rdd = orders_rdd.map(map_output)
        orders_rdd.saveAsTextFile(output)


if __name__ == "__main__":
    main()
