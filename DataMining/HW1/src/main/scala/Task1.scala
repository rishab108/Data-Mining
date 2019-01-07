import java.io.{File, PrintWriter}
import org.apache.spark.sql.SparkSession

object Task1 {

  def main(args: Array[String]) {

    val csv_file_read_path = args(0) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/survey_results_public.csv"
    val csv_file_write_path = args(1) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/Rishab_Kumar_task1.csv"

    val spark = SparkSession
      .builder()
      .appName("Task 1")
      .master("local[2]")
      .getOrCreate()

    val survey_df = spark.read.option("header", "true").csv(csv_file_read_path)
    val survey_rdd = survey_df.rdd

    val salary_response_rdd = survey_rdd
      .filter(row => row(52) != "NA" && row(52) != "0")
      .map(row => (row(3).asInstanceOf[String], 1))
      .reduceByKey(_ + _)
      .sortByKey()
      .cache()

    val total_rdd = salary_response_rdd
      .map(_._2)
      .sum()
      .toInt

    val result_array = salary_response_rdd
      .collect()

    val pw = new PrintWriter(new File(csv_file_write_path))

    pw.write("Total," + total_rdd + "\n")
    for (i <- result_array.indices) {
      pw.write("\"" + result_array(i)._1 + "\"," + result_array(i)._2 + "\n")
    }
    pw.close()

  }
}
