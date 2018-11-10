//importando librerias
import org.apache.spark.sql.SparkSession

// Optional: Utilizar el codigo de  Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Cree un sesion Spark
val spark = SparkSession.builder().getOrCreate()

//importando libreria de kmeans
import org.apache.spark.ml.clustering.KMeans

// Utilice Spark para leer el archivo csv Advertising.
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Wholesale_customers_data.csv")

// Imprime el schema en el DataFrame.
data.printSchema()

// Imprime un renglon de ejemplo del DataFrane.
data.head(1)

// Importe VectorAssembler y Vectors
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// imprime columnas
data.columns

//Se toman los datos de entrenamiento
val features_data = (data.select($"Region",$"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen"))

//// Creamos un nuevo objecto VectorAssembler llamado assembler para los feature
val assembler = (new VectorAssembler().setInputCols(Array("Fresh","Milk", "Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features"))

//trains the k-means model
val kmeans = new KMeans().setK(3).setSeed(1L)

val df = assembler.transform(features_data).select($"features")
val model = kmeans.fit(df)


// Evaluate clustering by calculate Within Set Sum of Squared Errors.
val WSSE = model.computeCost(df)
println(s"Within set sum of Squared Errors = $WSSE")

// Show results
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
