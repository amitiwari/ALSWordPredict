from pyspark import SparkContext
from pyspark.sql import SQLContext 
from pyspark import SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import sys, operator
import re, string, unicodedata
from pyspark.sql.functions import levenshtein, length
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

inputs = sys.argv[1]



conf = SparkConf().setAppName('recommend')
sc = SparkContext()
sqlContext = SQLContext(sc)



def func(x): return x

documents_rdd = sc.textFile(inputs)



splitdoc = documents_rdd.map(lambda line : (line.split(",")[0], (line.split(",")[2:])))

trainset = splitdoc.flatMapValues(func).map(lambda (docid,wordid) : (int(docid), int(wordid), 1.0)).cache()

rank = 10
numIterations = 10
lambdaunderscore = 0.1
alpha1 = 0.01
model = ALS.trainImplicit(trainset, rank, numIterations,lambda_ =0.1 , alpha = 0.1)
    
    
docid = documents_rdd.map(lambda line : int(line.split(",")[0])).collect()

mywords = []


    
givenwords = [line.strip() for line in open('/grad/1/amitt/CMPT732/DocumentWords.txt', 'r')]
     
resultwords = open('/grad/1/amitt/CMPT732/ResultWords.txt', 'w+')

for line in givenwords:
    docid = line.split(",")[0]
    resultwords.write(docid)
    resultwords.write(",")
    
    count = 0    
    
    for t in model.recommendProducts(int(docid), 5):
        if(count<4):
            resultwords.write(str(t[1]) + ",")
        else:
            resultwords.write(str(t[1]))
               
        count = count + 1 
        
    resultwords.write("\n")