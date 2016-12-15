from pyspark import SparkConf, SparkContext
import sys, operator, re, string , unicodedata
 
inputs = sys.argv[1]
output = sys.argv[2]


conf = SparkConf().setAppName('word count')
sc = SparkContext(conf=conf)
 
text = sc.textFile(inputs)
wordsep = re.compile(r'[%s\s]+' % re.escape(string.punctuation))
 
words = text.flatMap(lambda line: wordsep.split(line)).filter(lambda w : len(w)>0).map(lambda w: (unicodedata.normalize('NFD',w.lower()),1))
 
wordcount = words.reduceByKey(operator.add).coalesce(1).cache()
 
outdata = wordcount.sortBy(lambda (w,c): (-c,w)).map(lambda (w,c): u"%s %i" % (w, c))
outdata.saveAsTextFile(output + '/by-freq')

outdata1=wordcount.sortBy(lambda (w,c): w,1).map(lambda (w,c): u"%s %i" % (w, c))
outdata1.saveAsTextFile(output + '/by-word')
