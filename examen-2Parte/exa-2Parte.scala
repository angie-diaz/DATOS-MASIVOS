//paso 1 sIMPLE SESION EN SparkSession
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder().getOrCreate()

//paso 2 Cargar el archivo Netflix_2011_2016 stock.csv
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

//paso 3 Nombre de las columnas
df.columns

//paso 4 Esquema,Nombre de las columnas,tipo de dato y campo
df.printSchema()

//paso 5 Imprime las primeras 5 columnas
df.select("Date","Open","High","Low","Close").show()

//paso 6 Describe los datos de la columna
df.describe().show()

//paso 7 Crea un nuevo dataframe
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df("Volume")
df2.printSchema()
df2.show()

//paso Mostrar el dia que tuvo mas pique
df.groupBy("Date").agg(max("High")).show
df.orderBy("High".desc).show(1)

//paso 9 Significado de la columna cerrar
df.select("Close").describe().show()
println("refieren a los valores con la que cerró la bolsa de valores de Netflix")

//paso 10 Maximo y Minimo de la columna Volumen
df.select(max("High")).show()
df.select(min("High")).show()

//paso 11

//a) Cuantos dias fuel el cierre inferiror a $600
df.filter("Close > 600").count()

//b) Que porcentaje del tiempo fue el alto mayor de $500
(df.filter($"High" > 500).count() * 1.0/ df.count())*100

//c) Correlacion de pearson
df.select(corr("High","Volume")).show()

//d) Maximo alto por ano
val yeardf = df.withColumn("Year",year(df("Date")))
val yearmaxs = yeardf.select($"Year",$"High").groupBy("Year").max()
val res = yearmaxs.select($"Year",$"max(High)")
res.show()

//e) Cual es el promedio de cierre para cada mes del calendario
val monthdf = df.withColumn("Month",month(df("Date")))
val monthavgs = monthdf.select($"Month",$"Close").groupBy("Month").mean()
monthavgs.select($"Month",$"avg(Close)").orderBy("Month").show()
