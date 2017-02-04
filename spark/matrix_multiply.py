from pyspark import SparkConf, SparkContext
import sys, operator



def add_elements(a, b):
    return list(sum(p) for p in zip(a,b))
    
def multiply(row):
    rowData = []
    
    for element in row:
        for e in range(len(row)):
            rowData.append(float(element) * float(row[e]))

    
    return rowData

def main():


    inputs = sys.argv[1]
    output = sys.argv[2]

    conf = SparkConf().setAppName('Matrix Multiplication')
    sc = SparkContext(conf=conf)
    
    
    row = sc.textFile(inputs).map(lambda row : row.split(' ')).cache()
    total_columns = len(row.take(1)[0])
    #print total_columns
    
    mul_Result = row.map(multiply).reduce(add_elements)
    #print mul_Result
    #print len(mul_Result)
    fileOutput = open(output, 'w')
    
    
    final_Result = [mul_Result[x:x+10] for x in range(0, len(mul_Result), total_columns)]
    print final_Result
    
    for row in final_Result:
        for element in row:
            fileOutput.write(str(element) + ' ')
        fileOutput.write('\n')

    fileOutput.close()
    
    

if __name__ == "__main__":
    main()
