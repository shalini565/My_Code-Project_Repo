// Databricks notebook source

import org.apache.spark.sql.Encoders;

case class telecom(customerID: String, 
                   gender: String, 
                   SeniorCitizen: Int, 
                   Partner: String, 
                   Dependents: String, 
                   tenure: Int, 
                   PhoneService: String, 
                   MultipleLines: String, 
                   InternetService: String, 
                   OnlineSecurity: String, 
                   OnlineBackup: String, 
                   DeviceProtection: String, 
                   TechSupport: String, 
                   StreamingTV: String, 
                   StreamingMovies: String, 
                   Contract: String, 
                   PaperlessBilling: String, 
                   PaymentMethod: String, 
                   MonthlyCharges: Double, 
                   TotalCharges: Double, 
                   Churn: String )

val telecomSchema = Encoders.product[telecom].schema

val telecomDF = spark.read.schema(telecomSchema).option("header", "true").csv("/FileStore/tables/TelcoCustomerChurn.csv")

display(telecomDF)

// COMMAND ----------

// DBTITLE 1,Printing Schema
telecomDF.printSchema()

// COMMAND ----------

// DBTITLE 1,Creating Temp View from Dataframe 

telecomDF.createOrReplaceTempView("TelecomData")

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC #Exploratory Data Analysis

// COMMAND ----------

// DBTITLE 1,Customer Attrition in Data
// MAGIC %sql
// MAGIC
// MAGIC select Churn, count(Churn) from TelecomData group by Churn;

// COMMAND ----------

// DBTITLE 1,Gender Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select gender,count(gender), Churn  from TelecomData group by Churn,gender;
// MAGIC

// COMMAND ----------

// DBTITLE 1,SeniorCitizen Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select SeniorCitizen,count(SeniorCitizen), Churn  from TelecomData group by Churn,SeniorCitizen;
// MAGIC

// COMMAND ----------

// DBTITLE 1,Partner Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select Partner,count(Partner), Churn  from TelecomData group by Churn,Partner;
// MAGIC

// COMMAND ----------

// DBTITLE 1,Dependents Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select Dependents,count(Dependents), Churn  from TelecomData group by Churn,Dependents;

// COMMAND ----------

// DBTITLE 1,PhoneService Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select PhoneService,count(PhoneService), Churn  from TelecomData group by Churn,PhoneService;

// COMMAND ----------

// DBTITLE 1,MultipleLines Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select MultipleLines,count(MultipleLines), Churn  from TelecomData group by Churn,MultipleLines;

// COMMAND ----------

// DBTITLE 1,InternetService Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select InternetService,count(InternetService), Churn  from TelecomData group by Churn,InternetService;

// COMMAND ----------

// DBTITLE 1,OnlineSecurity Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select OnlineSecurity,count(OnlineSecurity), Churn  from TelecomData group by Churn,OnlineSecurity;

// COMMAND ----------

// DBTITLE 1,OnlineBackup Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select OnlineBackup,count(OnlineBackup), Churn  from TelecomData group by Churn,OnlineBackup;

// COMMAND ----------

// DBTITLE 1,DeviceProtection Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select DeviceProtection,count(DeviceProtection), Churn  from TelecomData group by Churn,DeviceProtection;

// COMMAND ----------

// DBTITLE 1,TechSupport Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select TechSupport,count(TechSupport), Churn  from TelecomData group by Churn,TechSupport;

// COMMAND ----------

// DBTITLE 1,StreamingTV Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select StreamingTV,count(StreamingTV), Churn  from TelecomData group by Churn,StreamingTV;

// COMMAND ----------

// DBTITLE 1,StreamingMovies Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select StreamingMovies,count(StreamingMovies), Churn  from TelecomData group by Churn,StreamingMovies;

// COMMAND ----------

// DBTITLE 1,Contract Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select Contract,count(Contract), Churn  from TelecomData group by Churn,Contract;

// COMMAND ----------

// DBTITLE 1,PaperlessBilling Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select PaperlessBilling,count(PaperlessBilling), Churn  from TelecomData group by Churn,PaperlessBilling;

// COMMAND ----------

// DBTITLE 1,PaymentMethod Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select PaymentMethod,count(PaymentMethod), Churn  from TelecomData group by Churn,PaymentMethod;

// COMMAND ----------

// DBTITLE 1,Tenure Group Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select cast ((TotalCharges/MonthlyCharges)/12 as Int) as Tenure, count(cast ((TotalCharges/MonthlyCharges)/12 as Int)), Churn  from TelecomData group by Churn,cast ((TotalCharges/MonthlyCharges)/12 as Int);

// COMMAND ----------

// DBTITLE 1,Tenure Group Distribution in Customer Attrition
// MAGIC %sql
// MAGIC
// MAGIC select cast ((TotalCharges/MonthlyCharges)/12 as Int) as Tenure, count(cast ((TotalCharges/MonthlyCharges)/12 as Int)) as counts, Churn  from TelecomData group by Churn,cast ((TotalCharges/MonthlyCharges)/12 as Int)  order by Tenure;

// COMMAND ----------

// DBTITLE 1,Monthly Charges & Total Charges by Tenure group
// MAGIC %sql
// MAGIC
// MAGIC select TotalCharges, MonthlyCharges, cast ((TotalCharges/MonthlyCharges)/12 as Int) as Tenure from TelecomData;

// COMMAND ----------

// MAGIC %md ## Creating a Classification Model
// MAGIC
// MAGIC In this Project, you will implement a classification model **(Logistic Regression)** that uses features of telecom details of customer and we will predict it is Churn or Not
// MAGIC
// MAGIC ### Import Spark SQL and Spark ML Libraries
// MAGIC
// MAGIC First, import the libraries you will need:

// COMMAND ----------

import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler

// COMMAND ----------


var StringfeatureCol = Array("customerID", "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn")

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
  new StringIndexer().setInputCol(colName).setHandleInvalid("skip").setOutputCol(colName + "_indexed")
}

val pipeline = new Pipeline()
                    .setStages(indexers)      

val TelDF = pipeline.fit(telecomDF).transform(telecomDF)

// COMMAND ----------

// DBTITLE 1,Printing Schema
TelDF.printSchema()

// COMMAND ----------

// DBTITLE 1,Data Display
TelDF.show()

// COMMAND ----------

// DBTITLE 1,Count of Records

TelDF.count()

// COMMAND ----------


val splits = TelDF.randomSplit(Array(0.7, 0.3))
val train = splits(0)
val test = splits(1)
val train_rows = train.count()
val test_rows = test.count()
println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

// COMMAND ----------


import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("customerID_indexed", "gender_indexed", "SeniorCitizen", "Partner_indexed", "Dependents_indexed", "PhoneService_indexed", "MultipleLines_indexed", "InternetService_indexed", "OnlineSecurity_indexed", "OnlineBackup_indexed", "DeviceProtection_indexed", "TechSupport_indexed", "StreamingTV_indexed", "StreamingMovies_indexed", "Contract_indexed", "PaperlessBilling_indexed", "PaymentMethod_indexed", "tenure", "MonthlyCharges", "TotalCharges" )).setOutputCol("features")
val training = assembler.transform(train).select($"features", $"Churn_indexed".alias("label"))
training.show()

// COMMAND ----------


import org.apache.spark.ml.classification.LogisticRegression

val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
val model = lr.fit(training)
println("Model Trained!")

// COMMAND ----------


val testing = assembler.transform(test).select($"features", $"Churn_indexed".alias("trueLabel"))
testing.show()

// COMMAND ----------

// MAGIC %md ### Test the Model
// MAGIC Now you're ready to use the **transform** method of the model to generate some predictions. But in this case you are using the test data which includes a known true label value, so you can compare the predicted Churn. 

// COMMAND ----------


val prediction = model.transform(testing)
val predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(200)

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

// MAGIC %md ### View the Raw Prediction and Probability
// MAGIC The prediction is based on a raw prediction score that describes a labelled point in a logistic function. This raw prediction is then converted to a predicted label of 0 or 1 based on a probability vector that indicates the confidence for each possible label value (in this case, 0 and 1). The value with the highest confidence is selected as the prediction.

// COMMAND ----------

prediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate=false)

// COMMAND ----------

// MAGIC %md
// MAGIC
// MAGIC Note that the results include rows where the probability for 0 (the first value in the probability vector) is only slightly higher than the probability for 1 (the second value in the probability vector). The default discrimination threshold (the boundary that decides whether a probability is predicted as a 1 or a 0) is set to 0.5; so the prediction with the highest probability is always used, no matter how close to the threshold.

// COMMAND ----------

// MAGIC %md ### Review the Area Under ROC
// MAGIC Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that you can use to compute this. The ROC curve shows the True Positive and False Positive rates plotted for varying thresholds.

// COMMAND ----------

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
val auc = evaluator.evaluate(prediction)
println("AUC = " + (auc))

// COMMAND ----------

// MAGIC %md ### Train a Naive Bayes Model
// MAGIC Naive Bayes can be trained very efficiently. With a single pass over the training data, it computes the conditional probability distribution of each feature given each label. For prediction, it applies Bayes’ theorem to compute the conditional probability distribution of each label given an observation.

// COMMAND ----------


import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("customerID_indexed", "gender_indexed", "SeniorCitizen", "Partner_indexed", "Dependents_indexed", "PhoneService_indexed", "MultipleLines_indexed", "InternetService_indexed", "OnlineSecurity_indexed", "OnlineBackup_indexed", "DeviceProtection_indexed", "TechSupport_indexed", "StreamingTV_indexed", "StreamingMovies_indexed", "Contract_indexed", "PaperlessBilling_indexed", "PaymentMethod_indexed", "tenure", "MonthlyCharges", "TotalCharges" )).setOutputCol("features")

val training = assembler.transform(TelDF).select($"features", $"Churn_indexed".alias("label"))

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = training.randomSplit(Array(0.9, 0.1), seed = 1234L)

// Train a NaiveBayes model.
val model = new NaiveBayes()
  .fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)

val predicted = predictions.select("features", "prediction", "label")
predicted.show()

// COMMAND ----------

// MAGIC %md ### Train a One-vs-Rest classifier (a.k.a. One-vs-All) Model
// MAGIC OneVsRest is an example of a machine learning reduction for performing multiclass classification given a base classifier that can perform binary classification efficiently. It is also known as “One-vs-All.”

// COMMAND ----------

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// load data file.
import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler().setInputCols(Array("customerID_indexed", "gender_indexed", "SeniorCitizen", "Partner_indexed", "Dependents_indexed", "PhoneService_indexed", "MultipleLines_indexed", "InternetService_indexed", "OnlineSecurity_indexed", "OnlineBackup_indexed", "DeviceProtection_indexed", "TechSupport_indexed", "StreamingTV_indexed", "StreamingMovies_indexed", "Contract_indexed", "PaperlessBilling_indexed", "PaymentMethod_indexed", "tenure", "MonthlyCharges", "TotalCharges" )).setOutputCol("features")

val training = assembler.transform(TelDF).select($"features", $"Churn_indexed".alias("label"))

// generate the train/test split.
val Array(train, test) = training.randomSplit(Array(0.8, 0.2))

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// train the multiclass model.
val ovrModel = ovr.fit(train)

// score the model on test data.
val predictions = ovrModel.transform(test)

val predicted = predictions.select("features", "prediction", "label")
predicted.show()


// COMMAND ----------

val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
