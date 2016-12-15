from pyspark import SparkConf, SparkContext
import sys, re, math

conf = SparkConf().setAppName('correlate logs')
sc = SparkContext(conf=conf)

def add_pairs(p1, p2):
    return (p1[0] + p2[0], p1[1]+p2[1])

def add_tuples(a, b):
    return tuple(sum(p) for p in zip(a,b))

inputs = sys.argv[1] 
output = sys.argv[2]

file_data = sc.textFile(inputs)

linere = re.compile("^(\\S+) - - \\[(\\S+) [+-]\\d+\\] \"[A-Z]+ (\\S+) HTTP/\\d\\.\\d\" \\d+ (\\d+)$")
KeyValue = file_data.map(lambda line : linere.split(line)).filter(lambda x : len(x)==6).map(lambda y : (y[1],(1, long(y[-2])))).reduceByKey(lambda a,b : add_pairs(a,b)).cache()
N = KeyValue.count()
ValSquare = KeyValue.map(lambda (k, (x,y)): (x,y, (x**2),(y**2), (x*y)))

ValSquare_sum = ValSquare.reduce(lambda a,b : add_tuples(a,b))

R= ((N*ValSquare_sum[4])-(ValSquare_sum[0]*ValSquare_sum[1]))/((math.sqrt((N*ValSquare_sum[2])-(ValSquare_sum[0]**2))*(math.sqrt((N*ValSquare_sum[3])-(ValSquare_sum[1]**2)))))

R_square = R**2

output_list = ["r= %r"%(R), "r^2 = %r" % (R_square), "Sum of x= %r"%(ValSquare_sum[0]),"Sum of Y= %r"%(ValSquare_sum[1]), "Sum of x^2= %r"%(ValSquare_sum[2]), "Sum of y^2= %r"%(ValSquare_sum[3]), "Sum of x*y= %r"%(ValSquare_sum[4])]
outdata = sc.parallelize(output_list).coalesce(1)

outdata.saveAsTextFile(output)
