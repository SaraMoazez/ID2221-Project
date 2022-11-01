import org.apache.spark.sql.cassandra._
import com.datastax.spark.connector._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.SaveMode

object Test{
//    def main(args: Array[String]) {
    def main() {

        //printf("innan spark")
        val spark = SparkSession.builder
            .appName("CompareMLAlgs")
            .master("local[2]")
            .getOrCreate()
        
        //printf("innan rawdata")
        val raw_data = spark.read.format("org.apache.spark.sql.cassandra").option("keyspace", "starfeatures").option("table", "data").load()
        
        //printf("innan show")
        //rawData.show  //For double-checking if the data was added and extracted correctly


        var number_of_features = 18
        val chosen_features = raw_data.columns.take(number_of_features)

        //discretize output
        val new_df =
            raw_data.withColumn("class", when(col("class") === "GALAXY", "0").otherwise(col("class")))
            .withColumn("class", when(col("class") === "STAR", "1").otherwise(col("class")))
            .withColumn("class", when(col("class") === "QSO", "2").otherwise(col("class")));
    
        //new_df.show()
 
        val features = chosen_features.filter((i: String) => i != "class")
        //features.foreach(println)
        val features2 = features.filter((i: String) => i != "obj_ID")


        val data_to_double = new_df.select(chosen_features.map(c => col(c).cast("double")): _*)
        //data_to_double.show()

        val train_test_split = data_to_double.randomSplit(Array(0.8, 0.2))
        val (training_data, test_data) = (train_test_split(0), train_test_split(1))
        //training_data.show()
        //test_data.show()

        // train models with x number of features 
        val assembler = new VectorAssembler().setInputCols(features2).setOutputCol("features")
        val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")

        // DT
        val dt = new DecisionTreeClassifier()
        val pipelineDT = new Pipeline().setStages(Array(assembler, labelIndexer, dt))
        val modelDT = pipelineDT.fit(training_data)

        val DTpredictions = modelDT.transform(test_data)
        val DTevaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val DTaccuracy = DTevaluator.evaluate(DTpredictions)
        println(s"Decision Tree Test Error = ${1.0 - DTaccuracy}")

        // RF
        val rf = new RandomForestClassifier()
        val pipelineRF = new Pipeline().setStages(Array(assembler, labelIndexer, rf))
        val modelRF = pipelineRF.fit(training_data)

        val RFpredictions = modelRF.transform(test_data)
        val RFevaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val RFaccuracy = RFevaluator.evaluate(RFpredictions)
        println(s"Random Forest Test Error = ${1.0 - RFaccuracy}")

        // LR
        val lr = new LogisticRegression()
        val pipelineLR = new Pipeline().setStages(Array(assembler, labelIndexer, lr))
        val modelLR = pipelineLR.fit(training_data)

        val LRpredictions = modelLR.transform(test_data)
        val LRevaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val LRaccuracy = LRevaluator.evaluate(LRpredictions)
        println(s"Logistic Regression Test Error = ${1.0 - LRaccuracy}")

        // MP
        val layers = Array[Int](17, 5, 5, 3)
        val mp = new MultilayerPerceptronClassifier().setLayers(layers)
        val pipelineMP = new Pipeline().setStages(Array(assembler, labelIndexer, mp))
        val modelMP = pipelineMP.fit(training_data)

        val MPpredictions = modelMP.transform(test_data)
        val MPevaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy")
        val MPaccuracy = MPevaluator.evaluate(MPpredictions)
        println(s"Multilayer Perceptron Test Error = ${1.0 - MPaccuracy}")
        

        DTpredictions.withColumnRenamed("obj_ID", "obj_id")
        RFpredictions.withColumnRenamed("obj_ID", "obj_id")
        LRpredictions.withColumnRenamed("obj_ID", "obj_id")
        MPpredictions.withColumnRenamed("obj_ID", "obj_id")

        val selected_predictions_DT = DTpredictions.select("obj_id", "prediction", "label")
        
        val selected_predictions_RF = RFpredictions.select("obj_id", "prediction", "label")

        val selected_predictions_LR = LRpredictions.select("obj_id", "prediction", "label")

        val selected_predictions_MP = MPpredictions.select("obj_id", "prediction", "label")

        selected_predictions_DT.write.format("org.apache.spark.sql.cassandra")
            .options(Map("keyspace"->"starfeatures","table"->"dtpredictions"))
            .mode(SaveMode.Overwrite)
            .save()
        val DT_data = spark.read.format("org.apache.spark.sql.cassandra").option("keyspace", "starfeatures").option("table", "dtpredictions").load()
        
        selected_predictions_RF.write.format("org.apache.spark.sql.cassandra")
            .options(Map("keyspace"->"starfeatures","table"->"rfpredictions"))
            .mode(SaveMode.Overwrite)
            .save()
        val RF_data = spark.read.format("org.apache.spark.sql.cassandra").option("keyspace", "starfeatures").option("table", "rfpredictions").load()
        
        selected_predictions_LR.write.format("org.apache.spark.sql.cassandra")
            .options(Map("keyspace"->"starfeatures","table"->"lrpredictions"))
            .mode(SaveMode.Overwrite)
            .save()
        val LR_data = spark.read.format("org.apache.spark.sql.cassandra").option("keyspace", "starfeatures").option("table", "lrpredictions").load()
        
        selected_predictions_MP.write.format("org.apache.spark.sql.cassandra")
            .options(Map("keyspace"->"starfeatures","table"->"mppredictions"))
            .mode(SaveMode.Overwrite)
            .save()
        val MP_data = spark.read.format("org.apache.spark.sql.cassandra").option("keyspace", "starfeatures").option("table", "mppredictions").load()
        
        DT_data.show()
        RF_data.show()
        LR_data.show()
        MP_data.show()


    }
}
