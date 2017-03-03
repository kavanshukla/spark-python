from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import json
import sys
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def main():
    conf = SparkConf().setAppName('Sentiment Analysis_Word2Vec')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    train_inputs = sys.argv[1]
    test_inputs = sys.argv[2]

    schema = StructType([
            StructField('reviewText', StringType(), False),
        StructField('overall', DoubleType(), False),
    ])

    
    read_json = sqlContext.read.json(train_inputs, schema)
    read_json.registerTempTable('read_json')
    lowercase = sqlContext.sql("""
    SELECT lower(reviewText) as reviewText, overall as label
    FROM read_json
    """)
    
    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")

    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    
    word2vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="features")
    
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    pipeline = Pipeline(stages=[regexTokenizer, remover, word2vec, lr])
    
    
    paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)

    cvModel = crossval.fit(lowercase)
        
    testDF = sqlContext.read.json(test_inputs, schema)
    testDF.registerTempTable('test_data')
    test_data = sqlContext.sql("""
    SELECT lower(reviewText) as reviewText, overall as label
    FROM test_data
    """)

    train_prediction = cvModel.transform(lowercase)
    test_prediction = cvModel.transform(test_data)
    evaluator = RegressionEvaluator()

    print "Training dataset RMSE error: %s" %str(evaluator.evaluate(train_prediction))
    print "Testing dataset RMSE: %s" %str(evaluator.evaluate(test_prediction))

if __name__ == "__main__":
    main()