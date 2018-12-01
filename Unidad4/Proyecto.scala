// Realizacion del proyecto de Datos Masivos
//Contenido del proyecto
//1.Objetivo: Comparacions del rendimiento de los siguientes algorimos de machine learning
//- K-means
//- Bisecting K-means
//- Con el dataset iris
//Las importaciones pertinentes para la visualizacion de los datos y manipulacion
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.clustering.BisectingKMeans

//Inicio de sesion en spark
val spark = SparkSession.builder().getOrCreate()
Logger.getLogger("org").setLevel(Level.ERROR)

//Cargar el archivo CSV para su lectura
val df = spark.read.option("inferSchema","true").csv("Iris.csv")
df.show()//visualizamos como vienen los datos

//Limpieza de datos, Considerando todos como vienen estos datos, para visualizarlo de una manera correcta
val CV = udf[Double, String](_.toDouble)
val df2 = df.withColumn("_c0",CV(df("_c0"))).withColumn("_c1",CV(df("_c1"))).
withColumn("_c2",CV(df("_c2"))).withColumn("_c3",CV(df("_c3"))).
select($"_c0".as("SepalLength"),$"_c1".as("SepalWidth"),$"_c2".as("PetalLength"),$"_c3".as("PetalWidth"))

df2.show()//visualizamos como quedan los datos despues de la limpieza

//// Creamos un nuevo objecto VectorAssembler llamado assembler para los feature
//Utilizacion de assembler y transformacion de los datos
val assembler = new VectorAssembler().setInputCols(Array("SepalLength","SepalWidth","PetalLength","PetalWidth")).setOutputCol("features")
val output = assembler.transform(df2).select($"features")
output.show()

//Utilizando K-KMeans
val kmeans = new KMeans().setK(5).setSeed(1L)
val model = kmeans.fit(output)

// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSE = model.computeCost(output)
println(s"Within set sum of Squared Errors = $WSSE")

// Show results
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

//COMPARATIVA CON ELANTERIOR//
//BISECTING KMEANS//////////////////////
///////////Utilizando Bisecting K-KMeans////
val bkm = new BisectingKMeans().setK(5).setSeed(1)
val model = bkm.fit(output)

val cost = model.computeCost(output)
println(s"Within Set Sum of Squared Errors = $cost")

println("Cluster Centers: ")
val centers = model.clusterCenters
centers.foreach(println)
