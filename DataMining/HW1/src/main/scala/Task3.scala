import java.io.{File, PrintWriter}

import org.apache.spark.sql.{Row, SparkSession}

object Task3 {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("Task 3")
      .master("local[2]")
      .getOrCreate()

    val csv_file_read_path = args(0) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/survey_results_public.csv"
    val csv_file_write_path = args(1) //"/Users/rishabkumar/Desktop/DM/HW_1/Data/Rishab_Kumar_task3.csv"

    val survey_df = spark.read.option("header", "true").csv(csv_file_read_path)
    val survey_rdd = survey_df.rdd

    val mapper = survey_rdd
      .filter(row => row(52) != "NA" && row(52) != "0")
      .map(row => (row(3).asInstanceOf[String], (1.0, getSalary(row), getSalary(row), getSalary(row))))
      .persist()

    val output = mapper.reduceByKey((v1, v2) => (v1._1 + v2._1, v1._2 + v2._2, getMin(v1._3, v2._3), getMax(v1._4, v2._4)))
      .mapValues(item => (item._1, item._3, item._4, item._2 / item._1))
      .sortByKey().collect()

    val pw = new PrintWriter(new File(csv_file_write_path))

    for (i <- output.indices) {
      pw.write("\"" + output(i)._1 + "\"," + output(i)._2._1.intValue() + "," + output(i)._2._2.intValue()
        + "," + output(i)._2._3.intValue() + "," + "%.2f".format(output(i)._2._4) + "\n")
    }
    pw.close()

  }

  def getSalary(row: Row) = {

    val salary = row(52).asInstanceOf[String]
      .replaceAll(",", "")
      .toDouble

    if (row(53) == "Monthly")
      salary * 12.0
    else if (row(53) == "Weekly")
      salary * 52.0
    else
      salary

  }

  def getMin(a: Double, b: Double): Double = {

    if (a > b)
      b
    else
      a
  }

  def getMax(a: Double, b: Double): Double = {

    if (a > b)
      a
    else
      b
  }
}
