import java.io.{File, PrintWriter}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import scala.math.min


object Rishab_Kumar_UserBasedCF {
  def main(args: Array[String]): Unit = {

    val start_time = System.currentTimeMillis()

    val neighborhood = 10000

    // paths for train and test file
    val train_data_csv_path = "/Users/rishabkumar/IdeaProjects/hw2/Data/train_review.csv" // args(0)
    val test_data_csv_path = "/Users/rishabkumar/IdeaProjects/hw2/Data/test_review.csv" //args(1)

    // setting the spark context
    val conf = new SparkConf().setAppName("UserBasedCF").setMaster("local[*]")
    val sc = new SparkContext(conf)

    // removing the log on output
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    // reading csv file and converting to RDD's
    val train_data = sc.textFile(train_data_csv_path)
    val test_data = sc.textFile(test_data_csv_path)

    // header removed from train data, data splitted and stored as RDD[Array[String]]
    val train_header = train_data.first()
    val train_data_split = train_data.filter(line => line != train_header)
      .map(_.split(","))
      .cache()

    // header removed from test data, data splitted and stored as RDD[Array[String]]
    val test_header = test_data.first()
    val test_data_split = test_data.filter(line => line != test_header)
      .map(_.split(","))
      .cache()

    // average rating for all items and users and is stored in this variable
    val train_ratings_average_tuple = train_data_split
      .map(item => (1, item(2).toDouble))
      .reduce((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
    val train_ratings_average = train_ratings_average_tuple._2 / train_ratings_average_tuple._1

    // average rating for each item is stored in this Scala Map
    val item_averages_map = train_data_split
      .map(item => (item(1), (item(2).toDouble, 1)))
      .reduceByKey((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
      .mapValues(v => v._1 / v._2)
      .collectAsMap()

    // average rating for each user is stored in this Scala map
    val user_averages_map = train_data_split
      .map(item => (item(0), (item(2).toDouble, 1)))
      .reduceByKey((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2))
      .mapValues(v => v._1 / v._2)
      .collectAsMap()

    // all item and user ratings are stored as a matrix in this, as Scala Map[String, Map[String, Double]]
    val ratings_matrix = train_data_split
      .map(item => (item(0), (item(1), item(2).toDouble)))
      .groupByKey()
      .sortByKey()
      .map { case (k, v) => (k, v.toMap) }
      .collectAsMap()

    // all users with the items set they have rated, it is a Scala Map[String, Set[String]]
    val clusters_map = train_data_split
      .map(item => (item(1), item(0)))
      .groupByKey()
      .map { case (k, v) => (k, v.toSet) }
      .collectAsMap()

    // this will store all the calculated similarities
    var similarities_matrix: Map[String, Map[String, Double]] = Map()

    // this map will store all the predictions for test data as a Scala Map(String, String), Double]
    var prediction_map: Map[(String, String), Double] = Map()

    // test data collected as Scala Array so that it can be traversed as a whole
    val test_data_array = test_data_split.collect()
    var size = test_data_array.length

    // this for loop traverses over all the test data points one by one and predict the rating for each
    for (test_data <- test_data_array) {

      val user = test_data(0)
      val item = test_data(1)
      val rating = test_data(2)

      // value to be predicted
      var prediction: Double = 0

      // if item is there in train data and user is also there in the train data, the simply compute the rating by formula
      if (user_averages_map.contains(user) && clusters_map.contains(item)) {

        var similarities_matrix_entry: Map[String, Double] = Map()

        // if the similarity map for the item is still not there, then we create new map else we retrieve the old map
        if (similarities_matrix.contains(user))
          similarities_matrix_entry = similarities_matrix(user)
        else
          similarities_matrix += (user -> similarities_matrix_entry)

        // we get the items that user has rated and iterate over those items only
        val item_cluster = clusters_map(item)
        // this is sum of rate*similarity for each item that user has rated
        var rate: Double = 0
        // it stores the sum for all the similarities
        var similarity_sum: Double = 0

        // stores all the neighbours with positive similarity and its similarity value
        var user_neighbors: Map[String, Double] = Map()

        // we iterate over those items only which user has rated, then find similarity of test item with those only
        for (user_inner <- item_cluster) {

          // stores the similarity value
          var value: Double = 0

          // we already have calculated similarity between test item and item in the user cluster, simply get that value
          if (similarities_matrix_entry.contains(user_inner))
            value = similarities_matrix_entry(user_inner)

          // when item pair is new and its similarity has not been calculated, we calculate it then
          else
            value = findSimilarity(user, ratings_matrix(user), user_inner, ratings_matrix(user_inner), user_averages_map)

          // if the similarity between two items is positive, then we add that item to its neighbors
          if (value > 0) {
            user_neighbors += (user_inner -> value)
          }
          // we store the calculated similarity in the similarity matrix for future use, irrespective of was it positive or negative
          similarities_matrix_entry += (user_inner -> value)
        }

        val user_neighbors_value_sorted = user_neighbors.toSeq.sortBy(_._2).reverse
        val user_neighbors_size = user_neighbors_value_sorted.length
        // iterate over top k neighbors and compute numerator and denominator
        for (i <- 0 until min(neighborhood, user_neighbors_size)) {

          rate += user_neighbors_value_sorted(i)._2 * (ratings_matrix(item)(user_neighbors_value_sorted(i)._1) - user_averages_map(user_neighbors_value_sorted(i)._1))
          similarity_sum += user_neighbors_value_sorted(i)._2
        }

        // check if the denominator in the rating formula is negative, then simply give item the rating as its average
        if (similarity_sum != 0) {
          prediction = user_averages_map(user) + (rate / similarity_sum)
          similarities_matrix += (user -> similarities_matrix_entry)
        } else {
          prediction = (item_averages_map(item) + user_averages_map(user) + train_ratings_average) / 3
        }
      }

      // when item exits in the train data but user is new, the simply rate item as the average rating of that item
      else if (user_averages_map.contains(user) && !clusters_map.contains(item)) {
        prediction = (user_averages_map(user) + train_ratings_average) / 2
      }

      // when item is new and never has been rated but user exists in the train data, them simply rate item as the average rating of the user
      else if (!user_averages_map.contains(user) && clusters_map.contains(item)) {
        prediction = (item_averages_map(item) + train_ratings_average) / 2
        // prediction = train_ratings_average
      }

      // when both item and user is new, the for now we will give it as the average rating of train data
      else {
        prediction = train_ratings_average
      }

      if (prediction > 5)
        prediction = 5.00

      // store the prediction into the scala map
      prediction_map += ((user, item) -> prediction)
    }

    // the prediction map gets transformed to Spark RDD to join it with original test data to find RMSE
    val prediction_rdd = sc.parallelize(prediction_map.toList)

    // test data transformed as RDD((User,Item), Rate) to calculate RMSE
    val test_data_true = test_data_split.map(item => ((item(0), item(1)), item(2).toDouble)).cache()

    // the original test data and predicted data gets joined as 1 RDD
    val ratesAndPreds = test_data_true.join(prediction_rdd)

    // MSE gets calculated for predictions
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = r1 - r2
      err * err
    }.mean()

    // it stores the square root of MSE calculated as RMSE
    val RMSE = math.sqrt(MSE)

    // store the absolute difference between rating and predictions
    val rateDiff = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      math.abs(r1 - r2)
    }

    // stores the count for all the ranges
    val range1 = rateDiff.filter { diff => diff >= 0 && diff < 1 }.count()
    val range2 = rateDiff.filter { diff => diff >= 1 && diff < 2 }.count()
    val range3 = rateDiff.filter { diff => diff >= 2 && diff < 3 }.count()
    val range4 = rateDiff.filter { diff => diff >= 3 && diff < 4 }.count()
    val range5 = rateDiff.filter { diff => diff >= 4 }.count()

    println(">=0 and <1:" + range1)
    println(">=1 and <2:" + range2)
    println(">=2 and <3:" + range3)
    println(">=3 and <4:" + range4)
    println(">=4:" + range5)

    println("RMSE :" + RMSE)

    // writes the sorted result into output file
    val outFileName = "src/main/Rishab_Kumar_UserBasedCF.txt"
    val pwrite = new PrintWriter(new File(outFileName))
    var output_list = ratesAndPreds.collect()
    output_list = output_list.sortWith(cmp)
    for (i <- output_list) {
      pwrite.write(i._1._1 + "," + i._1._2 + "," + i._2._2 + "\n")
    }
    pwrite.close()

    println("Time taken is " + (System.currentTimeMillis() - start_time) / 1000 + " secs")

  }

  // sort the result by user and then by business
  def cmp(e1: ((String, String), (Double, Double)), e2: ((String, String), (Double, Double))): Boolean = {
    var res: Boolean = false
    if (e1._1._1 != e2._1._1) {
      e1._1._1 < e2._1._1
    } else {
      e1._1._2 < e2._1._2
    }
  }

  // find the cosine similarity between two item vectors
  def findSimilarity(item_1_id: String, item_1: scala.collection.Map[String, Double],
                     item_2_id: String, item_2: scala.collection.Map[String, Double],
                     item_averages_map: scala.collection.Map[String, Double]): Double = {

    // find the common keys between two items to find there similarity
    val common_users = item_1.keySet.intersect(item_2.keySet)

    // if no keys are in common that return similarity as 0
    if (common_users.isEmpty)
      return 0

    // calculate numerators
    var dot_product: Double = 0
    for (item <- common_users)
      dot_product += (item_1(item) - item_averages_map(item_1_id)) * (item_2(item) - item_averages_map(item_2_id))

    var denominator_item_1: Double = 0
    var denominator_item_2: Double = 0

    // calculate denominators
    for ((k, v) <- item_1)
      denominator_item_1 += (v - item_averages_map(item_1_id)) * (v - item_averages_map(item_1_id))
    for ((k, v) <- item_2)
      denominator_item_2 += (v - item_averages_map(item_2_id)) * (v - item_averages_map(item_2_id))

    // divide both to get the similarity value
    val denominator = Math.sqrt(denominator_item_1) * Math.sqrt(denominator_item_2)
    if (denominator != 0)
      dot_product / denominator
    else
      0
  }
}
