import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.{BisectingKMeans, KMeans}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable.ListBuffer


object Task2 {
  def main(args: Array[String]): Unit = {

    val start_time = System.currentTimeMillis()

    val conf = new SparkConf().setAppName("Clustering").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val input_file_path = args(0)
    val input_data = sc.textFile(input_file_path)
    val algorithm = args(1)
    val clusters_num = args(2).toInt
    val iterations = args(3).toInt

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    var data_size = ""
    if (input_file_path.contains("small")) {
      data_size = "small"
    } else if (input_file_path.contains("large")) {
      data_size = "large"
    }

    val input_data_string_ = input_data.collect()
    val input_data_len = input_data_string_.length

    val unique_words = input_data
      .flatMap(_.split("\\s+"))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .map(item => item._1)
      .collect()
      .toSet

    val unique_words_len = unique_words.size

    var unique_word_index_map: Map[String, Int] = Map()
    var counter = 0
    for (word <- unique_words) {
      unique_word_index_map += (word -> counter)
      counter += 1
    }

    var idf_vector = new Array[Double](unique_words_len)
    val input_data_set = input_data
      .map(_.split("\\s+").toSet)
      .collect()

    for (i <- 0 until input_data_len) {
      val temp = input_data_set(i)
      for (word <- temp) {
        val index = unique_word_index_map(word)
        idf_vector(index) += 1
      }
    }
    idf_vector = idf_vector.map(x => math.log(input_data_len * 1.0 / x))
    val input_data_string = input_data
      .map(_.split("\\s+"))
      .collect()

    val input_data_vector_array = Array.ofDim[Double](input_data_len, unique_words_len)

    for (i <- 0 until input_data_len) {
      val temp = getTfVector(input_data_string(i), unique_word_index_map, unique_words_len)
      input_data_vector_array(i) = getTfidfVector(temp, idf_vector)
    }

    val tfidf: Array[Vector] = new Array[Vector](input_data_len)
    for (i <- 0 until input_data_len) {
      tfidf(i) = Vectors.dense(input_data_vector_array(i))
    }
    val tfidf_rdd = sc.parallelize(tfidf)

    var k_centroids = Array.ofDim[Double](clusters_num, unique_words_len)

    if (algorithm == "K") {
      val clusters = KMeans.train(tfidf_rdd, clusters_num, iterations, "k-means||", 42)
      val temp = clusters.clusterCenters
      for (i <- temp.indices) {
        k_centroids(i) = temp(i).toArray
      }

    }
    else if (algorithm == "B") {

      val bkm = new BisectingKMeans().setK(clusters_num).setMaxIterations(iterations).setSeed(42)
      val clusters = bkm.run(tfidf_rdd)
      val temp = clusters.clusterCenters
      for (i <- temp.indices) {
        k_centroids(i) = temp(i).toArray
      }

    }

    // this map stored each clusters ans review index which belongs to it
    var data_cluster_map: Map[Int, Set[Int]] = Map()
    for (i <- 0 until input_data_len) {

      var cluster_index = getClusterForReview(input_data_vector_array(i), k_centroids)
      if (data_cluster_map.contains(cluster_index)) {
        var temp = data_cluster_map(cluster_index)
        temp += i
        data_cluster_map += (cluster_index -> temp)
      } else {
        var temp: Set[Int] = Set()
        temp += i
        data_cluster_map += (cluster_index -> temp)
      }
    }

    var wsse: Double = 0
    var ids: ListBuffer[String] = new ListBuffer[String]()
    var sizes: ListBuffer[String] = new ListBuffer[String]()
    var errors: ListBuffer[String] = new ListBuffer[String]()
    var words: ListBuffer[String] = new ListBuffer[String]()

    for ((k, v) <- data_cluster_map) {
      var id = k + 1
      var size = v.size
      var error = getError(k_centroids(k), v, input_data_vector_array)
      var top_10_words = getTop10Words(v, input_data_string_, sc)
      wsse += error

      ids += id.toString
      sizes += size.toString
      errors += error.toString

      var word: StringBuilder = new StringBuilder
      for (item <- top_10_words) {
        word.append("\"" + item + "\",")
      }

      words += word.substring(0, word.size - 1).toString
    }

    var algotype = ""
    if (algorithm == "K") {
      algotype = "K-Means"
    } else if (algorithm == "B") {
      algotype = "Bisecting K-Means"
    }

    // making final ans
    var ans: StringBuilder = new StringBuilder
    ans.append("{\"algorithm\":\"" + algotype + "\", \"WSSE\":" + wsse + ", \"clusters\":[")
    for (i <- 0 until ids.size - 1) {
      ans.append("{" +
        "\"id\":" + ids(i) + "," +
        "\"size\":" + sizes(i) + "," +
        "\"error\":" + errors(i) + "," +
        "\"terms\":[" + words(i) + "," +
        "]},")
    }
    ans.append("{" +
      "\"id\":" + ids.last + "," +
      "\"size\":" + sizes.last + "," +
      "\"error\":" + errors.last + "," +
      "\"terms\":[" + words.last + "," +
      "]}")
    ans.append("]}")
    // final ans made

    var output_file_path = "Rishab_Kumar_Cluster_" + data_size + "_" + args(1) + "_" + args(2) + "_" + args(3) + ".json"
    var pw = new PrintWriter(new File(output_file_path))
    pw.write(ans.toString())
    pw.close()

    val end_time = System.currentTimeMillis()
    println("Time Taken is : " + (end_time - start_time) / 1000 + " secs")

  }

  def getTfidfVector(tf: Array[Double], idf: Array[Double]): Array[Double] = {
    var len = tf.length
    for (i <- 0 until len) {
      tf(i) *= idf(i)
    }
    tf
  }

  def getTfVector(review: Array[String], unique_word_index_map: Map[String, Int], unique_words_len: Int): Array[Double] = {

    var review_vector: Array[Double] = new Array(unique_words_len)
    var len = review.length

    for (word <- review) {
      var index = unique_word_index_map(word)
      review_vector(index) += 1
    }
    val max = review_vector.max
    review_vector = review_vector.map(x => x / len)
    //    review_vector = review_vector.map(x => x / max)

    review_vector

  }

  def getTop10Words(reviews_in_cluster_index: Set[Int], all_reviews: Array[String], sc: SparkContext): Array[String] = {

    var len = reviews_in_cluster_index.size
    var reviews_in_cluster_index_list = reviews_in_cluster_index.toList
    var reviews_in_cluster = new Array[String](len)
    for (i <- 0 until len) {
      var index = reviews_in_cluster_index_list(i)
      reviews_in_cluster(i) = all_reviews(index)
    }

    var reviews_in_cluster_rdd = sc.parallelize(reviews_in_cluster)
    var top_10_words = reviews_in_cluster_rdd
      .flatMap(_.split("\\s+"))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .sortBy(_._2, false)
      .map(item => item._1)
      .take(10)

    top_10_words
  }

  def getError(cluster_centroid: Array[Double], reviews_in_cluster_index: Set[Int], all_reviews: Array[Array[Double]]): Double = {

    var error: Double = 0
    var len = reviews_in_cluster_index.size
    var reviews_in_cluster_index_list = reviews_in_cluster_index.toList
    for (i <- 0 until len) {
      var index = reviews_in_cluster_index_list(i)
      var reviews_in_cluster = all_reviews(index)
      error += math.pow(computeEuclideanDistance(cluster_centroid, reviews_in_cluster), 2)
    }
    error
  }

  def getClusterForReview(review: Array[Double], k_centroids: Array[Array[Double]]): Int = {

    var min_distance = Double.MaxValue
    var min_distance_index = -1
    for (i <- k_centroids.indices) {

      var dist: Double = computeEuclideanDistance(review, k_centroids(i))
      if (dist < min_distance) {
        min_distance = dist
        min_distance_index = i
      }

    }
    min_distance_index
  }

  def computeEuclideanDistance(review: Array[Double], k_centroid: Array[Double]): Double = {

    var dist: Double = 0
    var len = review.length

    for (i <- 0 until len) {
      dist += math.pow(review(i) - k_centroid(i), 2)
    }
    dist = math.sqrt(dist)
    dist
  }
}
