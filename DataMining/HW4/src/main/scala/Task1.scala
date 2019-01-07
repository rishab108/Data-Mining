import java.io.{File, PrintWriter}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ListBuffer
import scala.util.Random


object Task1 {
  def main(args: Array[String]): Unit = {

    // to check the running time for algorithm
    val start_time = System.currentTimeMillis()
    // setting the spark context
    val conf = new SparkConf().setAppName("KMeans").setMaster("local[*]")
    val sc = new SparkContext(conf)
    // path for the input file and input which is RDD[String]
    val input_file_path = args(0)
    // args(0)
    val input_data = sc.textFile(input_file_path)
    // type of feature to use
    val feature = args(1)
    //args(1)
    // number of clusters
    val cluster = args(2).toInt
    // number of iterations
    val iterations = args(3).toInt
    // random number generator
    val random = new Random(seed = 42)

    // removing the log on output
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    var data_size = ""
    if (input_file_path.contains("small")) {
      data_size = "small"
    } else if (input_file_path.contains("large")) {
      data_size = "large"
    }

    // input data as Array[String] to get top words
    var input_data_string_ = input_data.collect()
    var input_data_len = input_data_string_.length

    // gets all unique words as dimension for vectors
    var unique_words = input_data
      .flatMap(_.split("\\s+"))
      .map(word => (word, 1))
      .reduceByKey(_ + _)
      .map(item => item._1)
      .collect()
      .toSet

    // total number of unique words
    var unique_words_len = unique_words.size

    // map for word and its index
    var unique_word_index_map: Map[String, Int] = Map()
    var counter = 0
    for (word <- unique_words) {
      unique_word_index_map += (word -> counter)
      counter += 1
    }

    // calculate idf vector for all documents
    var idf_vector = new Array[Double](unique_words_len)
    var input_data_set = input_data
      .map(_.split("\\s+").toSet)
      .collect()

    for (i <- 0 until input_data_len) {
      var temp = input_data_set(i)
      for (word <- temp) {
        var index = unique_word_index_map(word)
        idf_vector(index) += 1
      }
    }
    idf_vector = idf_vector.map(x => math.log(input_data_len * 1.0 / x))

    // input data of Array[Array[String]] to get tf-idf vectors
    var input_data_string = input_data
      .map(_.split("\\s+"))
      .collect()

    // convert input of form Array[Array[String]] to Array[Array[Double]] as its vectorized form
    var input_data_vector_array = Array.ofDim[Double](input_data_len, unique_words_len)
    if (feature == "T") {

      for (i <- 0 until input_data_len) {
        var temp = getTfVector(input_data_string(i), unique_word_index_map, unique_words_len)
        input_data_vector_array(i) = getTfidfVector(temp, idf_vector)
      }

    } else if (feature == "W") {

      for (i <- 0 until input_data_len) {
        input_data_vector_array(i) = getCountVector(input_data_string(i), unique_word_index_map, unique_words_len)
      }

    }

    // Now we have got out input in vectorized form, so now we start K-Means algorithm

    //initialize centroids for k clusters
    var k_centroids = Array.ofDim[Double](cluster, unique_words_len)
    for (i <- 0 until cluster) {
      var index = random.nextInt(input_data_len)
      k_centroids(i) = input_data_vector_array(index)
    }

    //    var k_centroids = sc.parallelize(input_data_vector_array).takeSample(true, cluster, seed = 42)

    // convert vector to rdd for spark
    var input_data_vector = sc.parallelize(input_data_vector_array)

    // compute centroids for k-means
    for (i <- 0 until iterations) {

      // get cluster a review belong to
      var k_centroids_new = input_data_vector
        .map(review => (getClusterForReview(review, k_centroids), review))
        .groupByKey()
        .map(item => (item._1, computeNewCluster(item._2.toArray)))
        .sortByKey()
        .map(item => item._2)
        .collect()

      k_centroids = k_centroids_new
    }

    // now we have got out K Centroids and now we compute which point belongs to which cluster

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

    // now this computes all the cluster size , error and top terms
    var algorithm = "K-Means"
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

    // making final ans
    var ans: StringBuilder = new StringBuilder
    ans.append("{\"algorithm\":\"K-Means\", \"WSSE\":" + wsse + ", \"clusters\":[")
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

    var output_file_path = "Rishab_Kumar_KMeans_" + data_size + "_" + args(1) + "_" + args(2) + "_" + args(3) + ".json"
    var pw = new PrintWriter(new File(output_file_path))
    pw.write(ans.toString())
    pw.close()

    val end_time = System.currentTimeMillis()
    println("Time Taken is : " + (end_time - start_time) / 1000 + " secs")

  }

  def getCountVector(review: Array[String], unique_word_index_map: Map[String, Int], unique_words_len: Int): Array[Double] = {

    var review_vector: Array[Double] = new Array(unique_words_len)

    for (word <- review) {
      var index = unique_word_index_map(word)
      review_vector(index) += 1
    }
    review_vector
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

  def computeNewCluster(data_for_k: Array[Array[Double]]): Array[Double] = {

    var len = data_for_k.length
    var len_each = data_for_k(0).length
    var new_centroid = new Array[Double](len_each)

    for (data_point <- data_for_k) {
      for (i <- 0 until len_each) {
        new_centroid(i) += data_point(i)
      }
    }
    new_centroid = new_centroid.map(x => x / len)
    new_centroid
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

}
