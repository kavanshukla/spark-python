from pyspark import SparkConf, SparkContext, SQLContext
import sys
import operator
import re, unicodedata, string
import pyspark_cassandra


def rows_to_list(key_vals, key_col, val_col, list_col, sqlContext):
        
        def listappend(lst, v):
                lst.append(v)
                return lst
        def listjoin(lst1, lst2):
                lst1.extend(lst2)
                return lst1


        assert key_vals.columns == [key_col, val_col], 'key_vals must have two columns: your key_col and val_col '
        key_val_rdd = key_vals.rdd.map(tuple)
        key_list_rdd = key_val_rdd.aggregateByKey([], listappend, listjoin)
        return sqlContext.createDataFrame(key_list_rdd, schema=[key_col, list_col])


def df_for_orders(keyspace, sc, sqlContext):

        table="orders"
        create_table=sc.cassandraTable(keyspace, table, split_size=10000)
        create_table=create_table.setName(table)
        return df_for(create_table, table, sqlContext)

def df_for_lineitem(keyspace, sc, sqlContext):

        table="lineitem"
        create_table=sc.cassandraTable(keyspace, table, row_format=pyspark_cassandra.RowFormat.DICT, split_size=10000).select("orderkey", "partkey")
        create_table=create_table.setName(table)
        return df_for(create_table, table, sqlContext)

def df_for_part(keyspace, sc, sqlContext):

        table="part"
        create_table=sc.cassandraTable(keyspace, table, row_format=pyspark_cassandra.RowFormat.DICT, split_size=10000).select("partkey", "name")
        create_table=create_table.setName(table)
        return df_for(create_table, table, sqlContext)

def df_for(create_table, tableName, sqlContext):

        df=sqlContext.createDataFrame(create_table)
        df.registerTempTable(tableName)
        return df

def map_primary_key(v):
        return v['orderkey'], v['clerk'], v['comment'], v['custkey'], v['order_priority'], v['orderdate'], v['orderstatus'], v['part_names'], v['ship_priority'], v['totalprice']

def main():
        input_keyspace = sys.argv[1]
        output_keyspace = sys.argv[2]

        cluster_seeds = ['199.60.17.136', '199.60.17.173']
        conf = SparkConf().set('spark.cassandra.connection.host', ','.join(cluster_seeds))
        conf.set('spark.dynamicAllocation.maxExecutors', 20)
        conf.setAppName('tpch orders denormalize cassandra')
        sc = pyspark_cassandra.CassandraSparkContext(conf=conf)
        sqlContext = SQLContext(sc)

        df_for_part(input_keyspace, sc, sqlContext)
        df_lineitems=df_for_lineitem(input_keyspace, sc, sqlContext)

        df_part=sqlContext.sql("""SELECT l.orderkey, p.name
                                  FROM lineitem l
                                  JOIN part p ON (l.partkey = p.partkey)""")



        df_part_names=rows_to_list(df_part, "orderkey", "name", "names", sqlContext)

        df_part_names.registerTempTable("part_names")


        df_orders=df_for_orders(input_keyspace, sc, sqlContext)


        df_orders=sqlContext.sql("""SELECT o.*, p.names as part_names
                                    FROM
                                    orders o
                                    JOIN part_names p ON (o.orderkey = p.orderkey) """)

        rdd_orders=df_orders.rdd.map(map_primary_key)

        rdd_orders.saveToCassandra(output_keyspace, 'orders_parts', parallelism_level=64)



if __name__ == "__main__":
        main()
