from pyspark import SparkConf, SparkContext, SQLContext
import sys
import operator
import re, unicodedata, string
import pyspark_cassandra


def df_for_orders(keyspace, sc, sqlContext, orderkeys):

        table="orders"
        create_table=sc.cassandraTable(keyspace, table, split_size=10000)
        create_table=create_table.where("orderkey in (" + ",".join(orderkeys) + ")")
        create_table=create_table.setName(table)
        return df_for(create_table, table, sqlContext)

def rdd_for_lineitem(keyspace, sc, sqlContext, orderkeys):

        table="lineitem"
        create_table=sc.cassandraTable(keyspace, table, row_format=pyspark_cassandra.RowFormat.DICT)
        create_table=create_table.where("orderkey in (" + ",".join(orderkeys) + ")")
        create_table=create_table.setName(table)
        create_table.cache()
        df_for(create_table, table, sqlContext)

        return df_for(create_table, table, sqlContext)

def df_for_part(keyspace, sc, sqlContext, lineitems):

        table="part"

        create_table=sc.cassandraTable(keyspace, table, row_format=pyspark_cassandra.RowFormat.DICT)
        create_table=create_table.where("partkey in (" + ",".join(lineitems) + ")")
        create_table=create_table.setName(table)
        return df_for(create_table, table, sqlContext)

def df_for(create_table, tableName, sqlContext):

        df=sqlContext.createDataFrame(create_table)
        df.registerTempTable(tableName)
        return df

def map_output(value):
        return 'Order #%s $%.2f: %s' % (value[0], value[1][0], value[1][1])

def reduce_parts(a, b):
        return (a[0], a[1]+", "+b[1])

def map_key(line):
        return (line[0], (line[1],line[2]))

def main():
        keyspace = sys.argv[1]
        output = sys.argv[2]
        orderkeys = sys.argv[3:]

        cluster_seeds = ['199.60.17.136', '199.60.17.173']
        conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds))
        conf.set('spark.dynamicAllocation.maxExecutors', 20)
        conf.setAppName('tpch orders in cassandra')
        sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
        sqlContext = SQLContext(sc)

        df_for_orders(keyspace, sc, sqlContext, orderkeys)
        lineitems=rdd_for_lineitem(keyspace, sc, sqlContext, orderkeys).collect()


        items=[]
        for item in lineitems:
                items.append("%s" % item['partkey'])

        df_for_part(keyspace, sc, sqlContext, items)



        df_orders=sqlContext.sql("""SELECT o.orderkey, o.totalprice, p.name FROM
                                    orders o
                                    JOIN lineitem l ON (o.orderkey = l.orderkey)
                                    JOIN part p ON (l.partkey = p.partkey)
                                    WHERE o.orderkey in (""" + ",".join(orderkeys) \
                                    +")")

        rdd_orders=df_orders.rdd
        rdd_orders=rdd_orders.map(map_key) \
                   .reduceByKey(reduce_parts)\
                   .sortByKey() \
                   .map(map_output).coalesce(1)
        rdd_orders.saveAsTextFile(output)

if __name__ == "__main__":
    main()
