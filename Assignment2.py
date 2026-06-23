from pyspark.sql import SparkSession
import pandas as pd


#Loading the income csv
spark = SparkSession.builder.appName('ml-income').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # hide the noisy WARN messages
df = spark.read.csv('income.csv',header = True, inferSchema = True)

# converting strings to numbers
from pyspark.ml.feature import StringIndexer, VectorAssembler
indexer = StringIndexer(inputCols=["workclass",'education','marital_status','occupation','relationship','race','sex','citizenship'], outputCols=["workclass_indx",'education_indx','marital_status_indx','occupation_indx','relationship_indx','race_indx','sex_indx','citizenship_indx'])
df = indexer.fit(df).transform(df)

labels = StringIndexer(inputCol = 'income_class', outputCol = 'label')
df = labels.fit(df).transform(df)

#creating feature column
numeric_columns = ['age', 'workclass_indx', 'weight', 'education_indx','education_years','marital_status_indx','occupation_indx','relationship_indx','race_indx','sex_indx','capital_gain','capital_loss','hours_per_week','citizenship_indx']
assembler = VectorAssembler(inputCols = numeric_columns,outputCol='features')
df = assembler.transform(df)
df.show()

#spliting the data
train, test = df.randomSplit([0.8,0.2], seed = 42)

print('Train size : '+str(train.count()))
print('Test size : '+ str(test.count()))

#defining the classifiers
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier

#maxBins = 50 is used as feature 13 has 42 unique values which exceeds the default of 32
rf = RandomForestClassifier(featuresCol='features', labelCol="label",maxBins=50) 
dt = DecisionTreeClassifier(featuresCol='features', labelCol="label",maxBins=50) 


#training and evaluating both the models
rfModel = rf.fit(train)
rf_predictions = rfModel.transform(test)
rf_predictions.select('age', 'workclass_indx', 'weight', 'education_indx','education_years','marital_status_indx','occupation_indx','relationship_indx','race_indx','sex_indx','capital_gain','capital_loss','hours_per_week','citizenship_indx', 'label', 'rawPrediction','prediction','probability').show(25)

dtModel = dt.fit(train)
dt_predictions = dtModel.transform(test)
dt_predictions.select('age', 'workclass_indx', 'weight', 'education_indx','education_years','marital_status_indx','occupation_indx','relationship_indx','race_indx','sex_indx','capital_gain','capital_loss','hours_per_week','citizenship_indx', 'label', 'rawPrediction','prediction','probability').show(25)

# Evaluating both models using accuracy metric
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction')
rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)

# Print accuracy for both models
print("Random Forest Classifier Accuracy: %s" % (rf_accuracy))
print('Random Forest Classifier Error: %s' % (1.0 - rf_accuracy))
print("Decision Tree Classifier Accuracy: %s" % (dt_accuracy))
print('Decision Tree Classifier Error: %s' % (1.0 - dt_accuracy))

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F

# Confusion matrix of both models
rf_preds = rf_predictions.select(['prediction', 'label']).withColumn('label', F.col('label').cast(FloatType()))
rf_preds = rf_preds.select(['prediction','label'])
metrics = MulticlassMetrics(rf_preds.rdd.map(tuple))
print('Random Forest Classifier confusion matrix:')
print(metrics.confusionMatrix().toArray())

print('Decision tree Classifier confusion matrix:')
dt_preds = dt_predictions.select(['prediction', 'label']).withColumn('label', F.col('label').cast(FloatType()))
dt_preds = dt_preds.select(['prediction','label'])
metrics = MulticlassMetrics(dt_preds.rdd.map(tuple))
print(metrics.confusionMatrix().toArray())