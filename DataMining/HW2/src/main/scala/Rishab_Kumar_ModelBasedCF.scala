import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, Logger}


object Rishab_Kumar_ModelBasedCF {
  def main(args: Array[String]): Unit = {

    var start_time = System.currentTimeMillis()

    // paths for train and test file
    val train_data_csv_path = "/Users/rishabkumar/IdeaProjects/hw2/Data/train_review.csv" // args(0)
    val test_data_csv_path = "/Users/rishabkumar/IdeaProjects/hw2/Data/test_review.csv" //args(1)

    // setting the spark context
    val conf = new SparkConf().setAppName("ModelBasedCF").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // removing the log on output
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    // reading csv file and converting to RDD's
    var train_data = sc.textFile(train_data_csv_path)
    var test_data = sc.textFile(test_data_csv_path)

    // header removed from train data, data splitted and stored as RDD[Array[String]]
    var train_header = train_data.first()
    train_data = train_data.filter(line => line != train_header).cache()

    // header removed from test data, data splitted and stored as RDD[Array[String]]
    var test_header = test_data.first()
    test_data = test_data.filter(line => line != test_header).cache()

    // average rating for all items and users and is stored in this variable
    val train_ratings_average_tuple = train_data
      .map(_.split(","))
      .map(item => (1, item(2).toDouble))
      .reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
    val train_ratings_average = train_ratings_average_tuple._2 / train_ratings_average_tuple._1

    // train and test data stored as Array[Array[String]]
    val train_data_array = train_data.map(_.split(",")).collect()
    val test_data_array = test_data.map(_.split(",")).collect()

    // map to store String id's as int for ALS model to run
    var user_map: Map[String, Int] = Map()
    var item_map: Map[String, Int] = Map()
    var user_counter = 0
    var item_counter = 0

    // Mapping user id to int
    for (entry <- train_data_array) {

      val user: String = entry(0)
      val item: String = entry(1)

      if (!user_map.contains(user)) {
        user_map += (user -> user_counter)
        user_counter += 1
      }

      if (!item_map.contains(item)) {
        item_map += (item -> item_counter)
        item_counter += 1
      }
    }

    // mapping item id to int
    for (entry <- test_data_array) {
      val user: String = entry(0)
      val item: String = entry(1)

      if (!user_map.contains(user)) {
        user_map += (user -> user_counter)
        user_counter += 1
      }

      if (!item_map.contains(item)) {
        item_map += (item -> item_counter)
        item_counter += 1
      }
    }

    // inverse maps to retrieve original id's back
    var user_map_inverse: Map[Int, String] = Map()
    var item_map_inverse: Map[Int, String] = Map()

    // reversing the user map
    for ((k, v) <- user_map) {
      if (!user_map_inverse.contains(v))
        user_map_inverse += (v -> k)
    }

    // reversing the item map
    for ((k, v) <- item_map) {
      if (!item_map_inverse.contains(v))
        item_map_inverse += (v -> k)
    }

    // converting train data to rating data for ALS
    val train_ratings = train_data.map(_.split(","))
      .map { case Array(user, item, rate) =>
        Rating(user_map(user), item_map(item), rate.toFloat)
      }

    // converting test data to rating data for ALS
    val test_ratings = test_data.map(_.split(","))
      .map { case Array(user, item, rate) =>
        Rating(user_map(user), item_map(item), rate.toFloat)
      }

    // it will store the predictions
    val test_predict = test_ratings.map { case Rating(user, product, rate) =>
      (user, product)
    }

    // setting the hyperparameters for our ALS model
    val rank = 10
    val numIterations = 10
    val model = ALS.train(train_ratings, rank, numIterations, 0.50, -1, 19L)

    // it stores all the predictions
    val predictions =
      model.predict(test_predict).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    // stores full test data
    val test_data_full = test_data
      .map(_.split(","))
      .map(item => ((user_map(item(0)), item_map(item(1))), train_ratings_average.toDouble))

    // joins test data with predictions to get missing
    val predictions_full = predictions.rightOuterJoin(test_data_full)

    // makes the final predictions by taking values from both
    val predictions_full_final = predictions_full
      .map(item => (item._1, {
        if (item._2._1.isEmpty)
          item._2._2
        else item._2._1.get
      }))

    // it stores the max and min predictions for scaling purpose
    val maxPred = predictions_full_final.map(_._2).max()
    val minPred = predictions_full_final.map(_._2).min()
    val scales = predictions_full_final.map(x => {
      var scaling = 5 * ((x._2 - minPred) / (maxPred - minPred)) + 1
      (x._1, scaling)
    })

    // joins actual and predicted ratings to compute rmse
    val ratesAndPreds = test_ratings.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(scales)


    // computing rmse
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()

    val RMSE = math.sqrt(MSE)
    val rateDiff = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      math.abs(r1 - r2)
    }

    // calculate the ranges for print
    var range1 = rateDiff.filter { diff => diff >= 0 && diff < 1 }.count()
    var range2 = rateDiff.filter { diff => diff >= 1 && diff < 2 }.count()
    var range3 = rateDiff.filter { diff => diff >= 2 && diff < 3 }.count()
    var range4 = rateDiff.filter { diff => diff >= 3 && diff < 4 }.count()
    var range5 = rateDiff.filter { diff => diff >= 4 }.count()

    println(">=0 and <1:" + range1)
    println(">=1 and <2:" + range2)
    println(">=2 and <3:" + range3)
    println(">=3 and <4:" + range4)
    println(">=4:" + range5)

    println("RMSE: " + RMSE)

    // writes the output
    val outFileName = "src/main/Rishab_Kumar_ModelBasedCF.csv"
    val pwrite = new PrintWriter(new File(outFileName))

    var outputList = ratesAndPreds.map(item => {
      (user_map_inverse(item._1._1), item_map_inverse(item._1._2), item._2._2)
    }).collect()

    outputList = outputList.sortWith(cmp)
    for (i <- outputList) {
      pwrite.write(i._1 + "," + i._2 + "," + i._3 + "\n")
    }
    pwrite.close()
    println("Time: " + (System.currentTimeMillis() - start_time) / 1000 + " secs")

  }

  // comparator to sort the output
  def cmp(e1: (String, String, Double), e2: (String, String, Double)): Boolean = {
    var res: Boolean = false
    if (e1._1 != e2._1) {
      e1._1 < e2._1
    } else {
      e1._2 < e2._2
    }
  }

}
