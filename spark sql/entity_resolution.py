from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import sys, re
import operator
from sets import Set
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.functions import udf

conf = SparkConf().setAppName('Entity_Resolution')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

Amazon_input = sys.argv[1]
Google_input = sys.argv[2]
Stopwords_file = sys.argv[3]
PerfectMapping_input = sys.argv[4]

def jaccard_similarity(j):
    r = re.search('\[(.*?)\],\[(.*?)\]', j)
    if r is None: return 0
    joinkey1 = Set(r.group(1).split(', '))
    joinkey2 = Set(r.group(2).split(', '))
    if len(joinkey1) == 0 or len(joinkey2) == 0: return 0

    common_tokens = [t for t in joinkey2 if t in joinkey1]
    combined_length = len(joinkey1) + len(joinkey2) - len(common_tokens)
    if combined_length == 0:
        return 0
    return float(len(common_tokens))/combined_length

def tokenize(r, stopwords):
    tokens = re.split('\W+', r)
    tokens = [t.lower() for t in tokens if t not in stopwords and t!= u'']
    return tokens

def token_distribution(r):
    id = r[0]
    token_str = r[1]
    m = re.search('\[(.*?)\]', token_str)
    tokens = m.group(1).split(', ')
    token_id_map = [(t, id) for t in tokens]

    return token_id_map

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()
        self.df1.show()
        self.df2.show()

    def preprocessDF(self, df, cols): 
        stopwords = self.stopWordsBC
        transform_udf = udf(lambda r: tokenize(r, stopwords))
        preprocessed_df = df.withColumn("joinKey", transform_udf(concat_ws(' ', df[cols[0]], df[cols[1]])))

        return preprocessed_df


    def filtering(self, df1, df2):
        sqlCt.registerDataFrameAsTable(df1, "df1")
        sqlCt.registerDataFrameAsTable(df2, "df2")
        flat_rdd1 = df1.select(df1.id, df1.joinKey).map(token_distribution).flatMap(lambda t: t)
        flat_rdd2 = df2.select(df2.id, df2.joinKey).map(token_distribution).flatMap(lambda t: t)

        flat_df1 = sqlCt.createDataFrame(flat_rdd1, ('token1', 'id1'))
        flat_df2 = sqlCt.createDataFrame(flat_rdd2, ('token2', 'id2'))
        sqlCt.registerDataFrameAsTable(flat_df1, "flat_df1")
        sqlCt.registerDataFrameAsTable(flat_df2, "flat_df2")

        joined_df = sqlCt.sql("""
                SELECT DISTINCT flat_df1.id1, flat_df2.id2 
                FROM flat_df2 JOIN flat_df1 
                ON (flat_df1.token1 = flat_df2.token2)       
                """)
        sqlCt.registerDataFrameAsTable(joined_df, "joined_df")
        
        new_df1 = sqlCt.sql("""
                SELECT joined_df.id1, df1.joinKey as joinKey1, joined_df.id2
                FROM joined_df JOIN df1 
                ON (df1.id = joined_df.id1)       
                """) 
        sqlCt.registerDataFrameAsTable(new_df1, "new_df1")
        
        new_df2 = sqlCt.sql("""
                SELECT new_df1.id1, new_df1.joinKey1, new_df1.id2, df2.joinKey as joinKey2
                FROM new_df1 JOIN df2 
                ON (df2.id = new_df1.id2)       
                """)

        return new_df2

    def verification(self, candDF, threshold):
        udf_jaccard = udf(lambda j: jaccard_similarity(j))
        jaccard_df = candDF.withColumn("jaccard", udf_jaccard(concat_ws(',', candDF.joinKey1, candDF.joinKey2)))
        return jaccard_df.where(jaccard_df.jaccard >= threshold)

    def evaluate(self, result, groundTruth):
        R_count = len(result)
        T_list = [t for t in result if t in groundTruth]
        T_count = float(len(T_list))
        precision = T_count/R_count

        A_count = len(groundTruth)
        recall = T_count/A_count

        fmeasure = 2*precision*recall/(precision+recall)

        return (precision, recall, fmeasure)

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        newDF1.show()
        newDF2.show()
        print "Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count()) 

        candDF = self.filtering(newDF1, newDF2)
        candDF.show()
        print "After Filtering: %d pairs left" %(candDF.count())

        resultDF = self.verification(candDF, threshold)
        print "After Verification: %d similar pairs" %(resultDF.count())
        resultDF.show()

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution(Amazon_input, Google_input, Stopwords_file)
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet(PerfectMapping_input) \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)