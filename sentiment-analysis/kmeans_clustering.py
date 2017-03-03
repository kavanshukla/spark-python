from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import json
import sys
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec, Word2VecModel, CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.clustering import KMeans

def getval(x,word_prediction):
    lists = []
    for i in range(0,len(x)):
        lists.append(str(word_prediction.value.get(x[i])))
    return lists
    

def main():
    conf = SparkConf().setAppName('Sentiment Analysis_KMeans')
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
    Tokens = regexTokenizer.transform(lowercase)
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    Removed_stop_words = remover.transform(Tokens)
    
    word2vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="features")
    model = word2vec.fit(Removed_stop_words)
    word_vectors = model.getVectors()
       
    word_vectors.registerTempTable('word_vectors')
    
    word_vectors_cols = sqlContext.sql("""
    SELECT word,vector as features
    FROM word_vectors
    """)
    
    vectors = sqlContext.sql("""
    SELECT vector as features
    FROM word_vectors
    """)
    
    kmeans = KMeans().setK(150).setSeed(1)
    kmeans_df = kmeans.fit(vectors)
    
    transformed = kmeans_df.transform(word_vectors_cols)
    transformed.registerTempTable('kmeans')
    wordpred = sqlContext.sql("""
        SELECT word,prediction
        FROM kmeans
    """)
    
    wordpred_rdd = wordpred.rdd
    wordpred_rdd = wordpred_rdd.map(lambda (word,prediction): (word,prediction))
    word_prediction = sc.broadcast(dict(wordpred_rdd.collect()))
    
    
    datta= Removed_stop_words.rdd
 
    wordss = datta.map(lambda c : (c[1],c[3]))
    
    pred_list = wordss.map(lambda (x,y): (x,getval(y,word_prediction)))
    
    countvectorizer_input = pred_list.toDF(['label','pred_vector'])
    cv = CountVectorizer(inputCol="pred_vector", outputCol="features")
    
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    pipeline = Pipeline(stages=[cv, lr])

    paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)

    lrModel = crossval.fit(countvectorizer_input)
    train_prediction = lrModel.transform(countvectorizer_input)
    evaluator = RegressionEvaluator()
    print "Training dataset RMSE error: %s" %str(evaluator.evaluate(train_prediction))
    
    #Testing dataset
    
    schema = StructType([
            StructField('reviewText', StringType(), False),
        StructField('overall', DoubleType(), False),
    ])

    
    read_json = sqlContext.read.json(test_inputs, schema)
    
    read_json.registerTempTable('read_json')
    lowercase = sqlContext.sql("""
    SELECT lower(reviewText) as reviewText, overall as label
    FROM read_json
    """)
    
    regexTokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")
    Tokens = regexTokenizer.transform(lowercase)
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    Removed_stop_words = remover.transform(Tokens)
    
    word2vec = Word2Vec(vectorSize=3, minCount=0, inputCol="filtered", outputCol="features")
    model = word2vec.fit(Removed_stop_words)
    word_vectors = model.getVectors()
    
    word_vectors.registerTempTable('word_vectors')
    
    word_vectors_cols = sqlContext.sql("""
    SELECT word,vector as features
    FROM word_vectors
    """)
    
    vectors = sqlContext.sql("""
    SELECT vector as features
    FROM word_vectors
    """)
    
    kmeans = KMeans().setK(150).setSeed(1)
    kmeans_df = kmeans.fit(vectors)
    
    transformed = kmeans_df.transform(word_vectors_cols)
    transformed.registerTempTable('kmeans')
    wordpred = sqlContext.sql("""
        SELECT word,prediction
        FROM kmeans
    """)
    
    wordpred_rdd = wordpred.rdd
    wordpred_rdd = wordpred_rdd.map(lambda (word,prediction): (word,prediction))
    word_prediction = sc.broadcast(dict(wordpred_rdd.collect()))
    
    datta= Removed_stop_words.rdd
    
    wordss = datta.map(lambda c : (c[1],c[3]))

    pred_list = wordss.map(lambda (x,y): (x,getval(y,word_prediction)))
    
    countvectorizer_input = pred_list.toDF(['label','pred_vector'])
    cv = CountVectorizer(inputCol="pred_vector", outputCol="features")
    
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    
    pipeline = Pipeline(stages=[cv, lr])
    paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=5)

    lrModel = crossval.fit(countvectorizer_input)
    test_prediction = lrModel.transform(countvectorizer_input)
    evaluator = RegressionEvaluator()
    print "Testing dataset RMSE error: %s" %str(evaluator.evaluate(test_prediction))
    
    
if __name__ == "__main__":
    main()