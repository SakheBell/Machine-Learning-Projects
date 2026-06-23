from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName('ml-iris').getOrCreate()
df = spark.read.csv('IRIS.csv',header = True, inferSchema = True)
df.printSchema()

#print(pd.DataFrame(df.take(5),columns=df.columns).transpose())

#print(df.dtypes)

numeric_features = [t[0] for t in df.dtypes if t[1]=='double']

from pyspark.ml.feature import VectorAssembler
 
numeric_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
assembler = VectorAssembler(inputCols = numeric_columns,outputCol='features')
df = assembler.transform(df)
#df.show()

from pyspark.ml.feature import StringIndexer
labels = StringIndexer(inputCol = 'species', outputCol = 'label')
df = labels.fit(df).transform(df)
df.show

print(pd.DataFrame(df.take(100), columns=df.columns))

train, test = df.randomSplit([0.7,0.3])

print('Train size : '+str(train.count()))
print('Test size : '+ str(test.count()))

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol="label")

rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label', 'rawPrediction','prediction','probability').show(25)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction')
accuracy = evaluator.evaluate(predictions)

print("Accuracy: %s" % (accuracy))
print('Error: %s' % (1.0 - accuracy))

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

preds = predictions.select(['prediction', 'label']).withColumn('label', F.col('label').cast(FloatType()))
preds = preds.select(['prediction','label'])
metrics = MulticlassMetrics(preds.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())