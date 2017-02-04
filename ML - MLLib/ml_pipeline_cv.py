from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


conf = SparkConf().setAppName("MLPipeline")
sc = SparkContext(conf=conf)

# Read training data as a DataFrame
sqlCt = SQLContext(sc)
trainDF = sqlCt.read.parquet("20news_train.parquet")

# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features", numFeatures=1000)
lr = LogisticRegression(maxIter=20, regParam=0.1)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])


paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [1000, 5000, 10000]) \
    .addGrid(lr.regParam, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) \
    .build()
    
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=2)  # use 3+ folds in practice

#cross validation
cvModel = crossval.fit(trainDF)

# Fit the pipeline to training data.
model = pipeline.fit(trainDF)

# Evaluate the model on testing data
testDF = sqlCt.read.parquet("20news_test.parquet")
prediction = model.transform(testDF)
prediction_cv = cvModel.transform(testDF)
evaluator = BinaryClassificationEvaluator()
print "areaUnderROC without parameter tuning: " + str(evaluator.evaluate(prediction))
print "areaUnderROC with parameter tuning: " + str(evaluator.evaluate(prediction_cv))
