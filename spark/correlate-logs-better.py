from pyspark import SparkConf, SparkContext
import sys, re, math

conf = SparkConf().setAppName('correlate-logs-better')
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

Total = KeyValue.map(lambda (k, (x,y)) : (1, (x,y)))
Total_final = Total.reduceByKey(lambda x,y : add_tuples(x,y)).collect()


Mean1 = Total_final[0][1][0]/float(N)
Mean2 = Total_final[0][1][1]/float(N)

ValSquare = KeyValue.map(lambda (k, (x,y)): (1,( (x-Mean1)**2, (y-Mean2)**2,(x-Mean1)*(y-Mean2)))).cache()

ValSquare_sum = ValSquare.reduceByKey(lambda a,b : add_tuples(a,b)).collect()
print ValSquare_sum
R = (ValSquare_sum[0][1][2])/((math.sqrt(ValSquare_sum[0][1][0]))*(math.sqrt(ValSquare_sum[0][1][1])))
R_square = float(R)**2

output_list = ["r= %r"%(float(R)), "r^2 = %r" % (float(R_square)), "Sum of x-Meanx= %r"%(ValSquare_sum[0][1][0]), "Sum of y-Meany= %r"%(ValSquare_sum[0][1][1])]
outdata = sc.parallelize(output_list).coalesce(1)

outdata.saveAsTextFile(output)
