import java.io.{File, PrintWriter}
import org.apache.spark.HashPartitioner
import org.apache.spark.sql.SparkSession

object Task2 {

  def main(args: Array[String]) {


    val csv_file_read_path = args(0) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/survey_results_public.csv"
    val csv_file_write_path = args(1) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/Rishab_Kumar_task2.csv"

    val spark = SparkSession
      .builder()
      .appName("Task 2")
      .master("local[2]")
      .getOrCreate()

    val survey_df = spark.read.option("header", "true").csv(csv_file_read_path)
    val survey_rdd = survey_df.rdd

    val salary_response_rdd = survey_rdd
      .filter(row => row(52) != "NA" && row(52) != "0")
      .map(row => (row(3).asInstanceOf[String], 1))
      .cache()

    // STANDARD RUN
    val start_time_standard = System.currentTimeMillis()

    val standard_partitioner = salary_response_rdd
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()

    val end_time_standard = System.currentTimeMillis() - start_time_standard

    // PARTITIONED RUN
    val start_time_partitioned = System.currentTimeMillis()

    val partitioned_rdd = salary_response_rdd
      .partitionBy(new HashPartitioner(2))
      .cache()

    val custom_partitioner = partitioned_rdd
      .reduceByKey(_ + _)
      .sortByKey()
      .collect()

    val end_time_partitioned = System.currentTimeMillis() - start_time_partitioned

    // WRITE DATA TO CSV

    val pw = new PrintWriter(new File(csv_file_write_path))

    val partition_items_without_partitioner = salary_response_rdd
      .mapPartitions(iterator => Iterator(iterator.size), true)
      .collect()

    pw.write("standard")
    for (item <- partition_items_without_partitioner) {
      pw.write("," + item)
    }
    pw.write("," + end_time_standard)


    val partition_items_with_partitioner = partitioned_rdd
      .mapPartitions(iterator => Iterator(iterator.size), true)
      .collect()

    pw.write("\n")

    pw.write("partition")
    for (item <- partition_items_with_partitioner) {
      pw.write("," + item)
    }
    pw.write("," + end_time_partitioned)

    pw.close()

  }

}

