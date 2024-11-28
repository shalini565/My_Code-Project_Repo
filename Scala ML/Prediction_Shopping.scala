// Databricks notebook source
val onlineshopperDF = sqlContext.read.format("csv")
  .option("header", "true")
  .option("inferSchema", "true")
  .option("delimiter", ",")
  .load("/FileStore/tables/online_shoppers_intention-245b7.csv")

display(onlineshopperDF)

// COMMAND ----------


onlineshopperDF.printSchema();

// COMMAND ----------


onlineshopperDF.select("Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend", "Revenue").describe().show()

// COMMAND ----------


onlineshopperDF.createOrReplaceTempView("WebData")

// COMMAND ----------

// DBTITLE 1,Temp
// MAGIC %sql
// MAGIC
// MAGIC select * from WebData;

// COMMAND ----------

// DBTITLE 1,Save it in Hive
import org.apache.spark.sql.hive.HiveContext
// sc - existing spark context
val sqlContext = new HiveContext(sc)
val df = sqlContext.sql("SELECT * FROM WebData")
df.coalesce(1).write.format("com.databricks.spark.csv").save("/data/home/sample.csv")

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC select Revenue as CustomerboughtProduct, count(Revenue) as counts  from WebData group by Revenue;

// COMMAND ----------

// DBTITLE 1,Displaying Percentage of Customer who bought Product (True/False)
// MAGIC %sql
// MAGIC
// MAGIC select Revenue as Customerwillbuy, count(Revenue) as counts  from WebData group by Revenue;

// COMMAND ----------

// DBTITLE 1,One Visualization to Rule Them All
// MAGIC %sql
// MAGIC
// MAGIC select Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend, Revenue from WebData;

// COMMAND ----------

// DBTITLE 1,Purchase on Weekends
// MAGIC %sql
// MAGIC
// MAGIC select Weekend, Count(Weekend) from WebData group by Weekend;

// COMMAND ----------

// DBTITLE 1,Types of Visitors
// MAGIC %sql
// MAGIC
// MAGIC select VisitorType, count(VisitorType) from WebData group by VisitorType

// COMMAND ----------

// DBTITLE 1,Types of Browser
// MAGIC %sql
// MAGIC
// MAGIC select Browser, Count(Browser) as BrowserType from WebData group by Browser;

// COMMAND ----------

// DBTITLE 1,Types of Traffic
// MAGIC %sql
// MAGIC
// MAGIC select TrafficType, count(TrafficType) from WebData group by TrafficType order by TrafficType;

// COMMAND ----------

// DBTITLE 1,Regions
// MAGIC %sql
// MAGIC
// MAGIC select Region, count(Region) from WebData group by Region;

// COMMAND ----------

// DBTITLE 1,Types of Operating Systems
// MAGIC %sql
// MAGIC
// MAGIC select OperatingSystems, count(OperatingSystems) from WebData group by OperatingSystems;

// COMMAND ----------

// DBTITLE 1,Months
// MAGIC %sql
// MAGIC
// MAGIC select Month, count(Month) from WebData group by Month;

// COMMAND ----------

// DBTITLE 1,Informational Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select Informational_Duration, Revenue from Webdata group by Informational_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,Administrative Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select Administrative_Duration,Revenue from Webdata group by Administrative_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,ProductRelated Duration VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select ProductRelated_Duration,Revenue from Webdata group by ProductRelated_Duration,Revenue;

// COMMAND ----------

// DBTITLE 1,Exit Rates VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC select ExitRates,Revenue from Webdata group by ExitRates,Revenue;

// COMMAND ----------

// DBTITLE 1,Page Values VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select PageValues,Revenue from Webdata group by PageValues,Revenue;

// COMMAND ----------

// DBTITLE 1,Bounce Rates VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select BounceRates,Revenue from Webdata group by BounceRates,Revenue;

// COMMAND ----------

// DBTITLE 1,Type of Traffic VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select TrafficType, count(TrafficType), Revenue from Webdata group by TrafficType,Revenue;

// COMMAND ----------

// DBTITLE 1,Visitor Type VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select VisitorType, count(VisitorType), Revenue from Webdata group by VisitorType,Revenue;

// COMMAND ----------

// DBTITLE 1,Region Type VS Revenue
// MAGIC %sql
// MAGIC
// MAGIC
// MAGIC select Region, count(Region), Revenue from Webdata group by Region,Revenue;

// COMMAND ----------

// DBTITLE 1,Administrative VS Informational
// MAGIC %sql
// MAGIC
// MAGIC select Administrative, Informational from WebData;

// COMMAND ----------

// MAGIC %md ## Creating a Logistic Regression Model
// MAGIC
// MAGIC In this Project, you will implement a Linear regression model that will Prediction Online Shopper Purchase Intention based on many attributes available in Website and Customer Data
// MAGIC
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC
// MAGIC First, import the libraries you will need:

// COMMAND ----------

// DBTITLE 1,Importing Apache Spark Library

import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------

// MAGIC %md ### Prepare the Training Data
// MAGIC
// MAGIC To train the regression model, you need a training data set that includes a vector of numeric features, and a label column. In this project, you will use the **VectorAssembler** class to transform the feature columns into a vector, and then rename the **revenue** column to **label**.

// COMMAND ----------

// MAGIC %md ###VectorAssembler()
// MAGIC
// MAGIC VectorAssembler():  is a transformer that combines a given list of columns into a single vector column. It is useful for combining raw features and features generated by different feature transformers into a single feature vector, in order to train ML models like logistic regression and decision trees. 
// MAGIC
// MAGIC **VectorAssembler** accepts the following input column types: **all numeric types, boolean type, and vector type.** 
// MAGIC
// MAGIC In each row, the **values of the input columns will be concatenated into a vector** in the specified order.

// COMMAND ----------

// DBTITLE 1,List all String Data Type Columns in an Array in further processing

var StringfeatureCol = Array("Month", "VisitorType")

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC ###StringIndexer
// MAGIC
// MAGIC StringIndexer encodes a string column of labels to a column of label indices.

// COMMAND ----------

// DBTITLE 1,Example of StringIndexer
import org.apache.spark.ml.feature.StringIndexer

val df = spark.createDataFrame(
  Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
).toDF("id", "category")

val indexer = new StringIndexer()
  .setInputCol("category")
  .setOutputCol("categoryIndex")

val indexed = indexer.fit(df).transform(df)

display(indexed)

// COMMAND ----------

// MAGIC %md ### Define the Pipeline
// MAGIC A predictive model often requires multiple stages of feature preparation. 
// MAGIC
// MAGIC A pipeline consists of a series of *transformer* and *estimator* stages that typically prepare a DataFrame for modeling and then train a predictive model. 
// MAGIC
// MAGIC In this case, you will create a pipeline with stages:
// MAGIC
// MAGIC - A **StringIndexer** estimator that converts string values to indexes for categorical features
// MAGIC - A **VectorAssembler** that combines categorical features into a single vector

// COMMAND ----------


import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}

val indexers = StringfeatureCol.map { colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
}

val pipeline = new Pipeline()
                    .setStages(indexers)      

val ShopperDF = pipeline.fit(onlineshopperDF).transform(onlineshopperDF)

// COMMAND ----------


ShopperDF.printSchema()

// COMMAND ----------


ShopperDF.show()

// COMMAND ----------

// DBTITLE 1,Converting Boolean Value to Integer since Model cannot process Boolean Value

val FinalShopperDF = ShopperDF
  .withColumn("RevenueInt",$"Revenue".cast("Int"))
  .withColumn("WeekendInt",$"Weekend".cast("Int"))

// COMMAND ----------


FinalShopperDF.show()

// COMMAND ----------

// MAGIC %md ### Split the Data
// MAGIC It is common practice when building supervised machine learning models to split the source data, using some of it to train the model and reserving some to test the trained model. In this project, you will use 70% of the data for training, and reserve 30% for testing. In the testing data, the **label** column is renamed to **trueLabel** so you can use it later to compare predicted labels with known actual values.

// COMMAND ----------


val splits = FinalShopperDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------

// DBTITLE 1,VectorAssembler() that combines categorical features into a single vector

val assembler = new VectorAssembler().setInputCols(Array("Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType_indexed", "WeekendInt", "RevenueInt")).setOutputCol("features")

val training = assembler.transform(train).select($"features", $"RevenueInt".alias("label"))

training.show(false)

// COMMAND ----------

// MAGIC %md ### Train a Regression Model
// MAGIC Next, you need to train a regression model using the training data. To do this, create an instance of the regression algorithm you want to use and use its **fit** method to train a model based on the training DataFrame. In this Project, you will use a *Logistic Regression* algorithm - though you can use the same technique for any of the Linear regression algorithms supported in the spark.ml API.

// COMMAND ----------

import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
val model = lr.fit(training)
println("Model Trained!")

// COMMAND ----------

// MAGIC %md ### Prepare the Testing Data
// MAGIC Now that you have a trained model, you can test it using the testing data you reserved previously. First, you need to prepare the testing data in the same way as you did the training data by transforming the feature columns into a vector. This time you'll rename the **revenue** column to **trueLabel**.

// COMMAND ----------


val testing = assembler.transform(test).select($"features", $"RevenueInt".alias("trueLabel"))
testing.show(false)

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Revenue. 

// COMMAND ----------


val prediction = model.transform(testing)
val predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()

// COMMAND ----------

// MAGIC %md Looking at the result, the **prediction** column contains the predicted value for the label, and the **trueLabel** column contains the actual known value from the testing data. It looks like there is some variance between the predictions and the actual values (the individual differences are referred to as *residuals*) you'll learn how to measure the accuracy of a model.

// COMMAND ----------

// MAGIC %md ### Compute Confusion Matrix Metrics
// MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
// MAGIC - True Positives
// MAGIC - True Negatives
// MAGIC - False Positives
// MAGIC - False Negatives
// MAGIC
// MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.

// COMMAND ----------


val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
  val metrics = spark.createDataFrame(Seq(
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn)))).toDF("metric", "value")
metrics.show()

// COMMAND ----------

// MAGIC %md ### Classification model Evaluation
// MAGIC
// MAGIC spark.mllib comes with a number of machine learning algorithms that can be used to learn from and make predictions on data. When these algorithms are applied to build machine learning models, there is a need to evaluate the performance of the model on some criteria, which depends on the application and its requirements. spark.mllib also provides a suite of metrics for the purpose of evaluating the performance of machine learning models.

// COMMAND ----------

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("trueLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(prediction)
