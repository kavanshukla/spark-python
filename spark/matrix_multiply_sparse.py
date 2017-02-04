from pyspark import SparkConf, SparkContext
import sys, operator
from scipy import *
from scipy.sparse import csr_matrix


def create_CSR_Matrix(inputs):
    row = []
    col = []
    data = []

    for values in inputs:
        value = values.split(':')
        row.append(0)
        col.append(int(value[0]))
        data.append(float(value[1]))
    
    return csr_matrix((data,(row,col)), shape=(1,100))

def multiply_Matrix(csrMatrix):

    csrTransponse = csrMatrix.transpose(copy=True)

    return (csrTransponse*csrMatrix)


def formatting(indexwithvalue):
    return ' '.join(map(lambda pair : str(pair[0]) + ':' + str(pair[1]), indexwithvalue))


def main():


    inputs = sys.argv[1]
    output = sys.argv[2]

    conf = SparkConf().setAppName('Sparse Matrix Multiplication')
    sc = SparkContext(conf=conf)
    
    sparseMatrix = sc.textFile(inputs).map(lambda row : row.split(' ')).map(create_CSR_Matrix).map(multiply_Matrix).reduce(operator.add)
    outputFile = open(output, 'w')
        
    print sparseMatrix
    
    for row in range(len(sparseMatrix.indptr)-1):
        column = sparseMatrix.indices[sparseMatrix.indptr[row]:sparseMatrix.indptr[row+1]]
        data = sparseMatrix.data[sparseMatrix.indptr[row]:sparseMatrix.indptr[row+1]]
        indexwithvalue = zip(column,data)
        final_format = formatting(indexwithvalue)
        outputFile.write(final_format + '\n')
