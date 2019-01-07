import java.io.{File, PrintWriter}
import org.apache.spark.{SparkConf, SparkContext}


object Rishab_Kumar_SON {
  def main(args: Array[String]): Unit = {

    // to check the running time for algorithm
    val start_time = System.currentTimeMillis()
    // setting the spark context
    val conf = new SparkConf().setAppName("SON").setMaster("local[*]")
    val sc = new SparkContext(conf)
    // path for the input file and input which is RDD[String]
    val input_file_path = args(0)
    val input_data = sc.textFile(input_file_path)

    // support threshold
    val support_threshold: Int = args(1).toInt

    // path of file to write final output
    val output_file_path = args(2)
    val pw = new PrintWriter(new File(output_file_path))

    // creates a RDD of Set[items in basket], re-partition it into 10 partitions and cache it
    val basket_items_map_rdd = input_data
      .map(_.split(","))
      .map(array => (array(0), array(1)))
      .groupByKey()
      .mapValues(map_value => map_value.toSet)
      .map { case (k, v) => v }
      .cache()

    // partition size
    val repartition_size: Double = basket_items_map_rdd.getNumPartitions

    // Map task output of phase 1
    val map_phase_1 = basket_items_map_rdd
      .mapPartitions(partition => Apriori(partition.toList, support_threshold, repartition_size))
      .map(item => (item, 1))
      .cache()

    // Reduce task output of phase 1
    val reduce_phase_1 = map_phase_1
      .reduceByKey(_ + _)
      .map { case (k, v) => k }
      .collect()

    // Map task output of phase 2
    val map_phase_2 = basket_items_map_rdd
      .mapPartitions(partition => GetFrequentItemSetSupport(partition.toList, reduce_phase_1.toSet))
      .cache()

    // Reduce task output phase 2, sort set with its size
    val reduce_phase_2 = map_phase_2
      .reduceByKey(_ + _)
      .filter(item => item._2 >= support_threshold)
      .map { case (k, v) => k }
      .cache()


    // generate final output
    val final_output = reduce_phase_2
      .map(item => (item.size, item.toList.sorted))
      .groupByKey()
      .sortByKey()
      .mapValues(item => sort(item.toList))
      .map { case (k, v) => v }
      .collect()

    // write output to final file
    for (item <- final_output) {
      var item_str = item.mkString(", ")
      item_str = item_str.replace("List", "")
      pw.write(item_str)
      pw.write("\n\n")
    }

    pw.close()

    // calculate total run time for the algorithm
    val end_time = System.currentTimeMillis()
    println("Time Taken : " + (end_time - start_time) / 1000 + " sec")

  }

  // runs Apriori algorithm on each chunk of data i.e. eqch partition
  def Apriori(partition: List[Set[String]], support_threshold: Double, repartition_size: Double): Iterator[Set[String]] = {

    // contains final list of frequent item sets including singletons, pairs, triplets etc
    var final_frequent_items_set: Set[Set[String]] = Set()
    // stores count of each item and its count to filter out singletons and returns the set of singletons in a List
    var frequent_items_set_singeltons: Set[Set[String]] = getSingletonSet(partition, support_threshold, repartition_size)
    //add the singleton sets to final list
    final_frequent_items_set ++= frequent_items_set_singeltons
    // stores frequent item sets of size k, which is initially singletons at start, to generate pairs
    var frequent_items_sets_of_size_k: Set[Set[String]] = frequent_items_set_singeltons
    // all item sets of given size without filtering
    var all_items_sets_of_size_k: Set[Set[String]] = Set()
    // to get the size of set
    var K = 2

    while (frequent_items_sets_of_size_k.nonEmpty) {
      // gets all item sets of size k
      all_items_sets_of_size_k = getAllItemSetsOfSizeK(frequent_items_sets_of_size_k, K)
      // filters out all item sets of size k which are greater than support threshold
      frequent_items_sets_of_size_k = getFrequentItemSets(partition, all_items_sets_of_size_k, support_threshold, repartition_size)
      // if new set is not empty add to final list
      final_frequent_items_set ++= frequent_items_sets_of_size_k
      // increment the size of frequent item sets you want
      K += 1
    }
    // returns final frequent item sets
    final_frequent_items_set.toIterator
  }

  // read all baskets and get the singletons sets which are frequent
  def getSingletonSet(partition: List[Set[String]], support_threshold: Double, repartition_size: Double): Set[Set[String]] = {

    var singleton_map: Map[String, Int] = Map()
    var frequent_singleton_set: Set[Set[String]] = Set()

    for (set <- partition) {
      for (item <- set) {

        if (singleton_map.contains(item)) {
          var count = singleton_map(item) + 1
          singleton_map += (item -> count)
        } else {
          singleton_map += (item -> 1)
        }
      }
    }

    for ((k, v) <- singleton_map) {

      if (v >= (support_threshold * 1.0 / repartition_size)) {

        var singleton_set: Set[String] = Set(k)
        frequent_singleton_set += singleton_set
      }
    }
    frequent_singleton_set
  }

  // get all sets of size k by joining singletons with sets of size k-1
  def getAllItemSetsOfSizeK(frequent_items_set_K: Set[Set[String]], K: Int): Set[Set[String]] = {

    var all_item_sets_of_size_k: Set[Set[String]] = Set()
    for (frequent_item <- frequent_items_set_K) {
      for (frequent_item_inner <- frequent_items_set_K) {
        var temp = frequent_item.union(frequent_item_inner)
        if (temp.size == K)
          all_item_sets_of_size_k += temp
      }
    }
    all_item_sets_of_size_k
  }

  // filters out all those frequent item sets which are greater than threshold
  def getFrequentItemSets(partition: List[Set[String]], all_combinations: Set[Set[String]], support_threshold: Double, repartition_size: Double): Set[Set[String]] = {

    var items_sets_map: Map[Set[String], Int] = Map()
    for (set <- all_combinations) {
      var count = 0
      for (bucket <- partition) {
        if (set.subsetOf(bucket))
          count += 1
      }
      items_sets_map += (set -> count)
    }

    var frequent_items_set: Set[Set[String]] = Set()
    for ((k, v) <- items_sets_map) {

      if (v >= (support_threshold * 1.0 / repartition_size)) {
        frequent_items_set += k
      }
    }
    frequent_items_set
  }

  // combine all and get final frequent item sets
  def GetFrequentItemSetSupport(partition: List[Set[String]], reduce_phase_1: Set[Set[String]]): Iterator[(Set[String], Int)] = {

    var frequent_item_set_support_map: Map[Set[String], Int] = Map()
    for (frequent_item_set <- reduce_phase_1) {
      for (basket <- partition) {

        if (frequent_item_set.subsetOf(basket)) {

          if (!frequent_item_set_support_map.contains(frequent_item_set)) {
            frequent_item_set_support_map += (frequent_item_set -> 1)
          } else {
            var count = frequent_item_set_support_map(frequent_item_set) + 1
            frequent_item_set_support_map += (frequent_item_set -> count)
          }

        }

      }
    }
    frequent_item_set_support_map.toIterator
  }

  // sort List[List[String]] lexographically
  def sort[A](coll: Seq[Iterable[A]])(implicit ordering: Ordering[A]) = coll.sorted
}
