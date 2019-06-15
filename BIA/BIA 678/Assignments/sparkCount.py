import pyspark

sc = pyspark.SparkContext('local[*]')
rdd = sc.parallelize(range(20))
print(rdd.takeSample(False,5))