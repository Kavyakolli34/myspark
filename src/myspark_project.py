
# 1 Start Spark Session

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("HotelBookingProject") \
    .master("local[*]") \
    .enableHiveSupport() \
    .getOrCreate()

print("Spark Session started:", spark.version)


# 2 Load Raw Data from CSV

file_path = "/tmp/DE011025/kavya/hotel_bookings/hotel_bookings.csv"  

df_raw = spark.read.csv(file_path, header=True, inferSchema=True)
print("Total rows:", df_raw.count())
df_raw.show(5)
df_raw.printSchema()


# 3 Basic Validation

print("Columns:", len(df_raw.columns))
df_raw.describe().show()


# 4 Missing Value Check

from pyspark.sql.functions import col, sum as _sum, when

missing_summary = df_raw.select([
    (_sum(when(col(c).isNull(), 1).otherwise(0)) / df_raw.count() * 100).alias(c)
    for c in df_raw.columns
])
missing_summary.show(vertical=True)


# 5 Data Cleaning

from pyspark.sql.functions import trim

# Drop Duplicates
df_clean = df_raw.dropDuplicates()

# Trim Spaces in String Columns
df_clean = df_clean.select([
    trim(col(c)).alias(c) if str(df_clean.schema[c].dataType) == "StringType" else col(c)
    for c in df_clean.columns
])

# Fill Nulls
df_clean = df_clean.fillna({
    "country": "Unknown",
    "children": 0,
    "babies": 0,
    "agent": 0,
    "company": 0
})

# Drop rows missing key fields
df_clean = df_clean.dropna(subset=["hotel", "arrival_date_year", "adr"])

print("After cleaning:", df_clean.count())
df_clean.show(5)


# 6 Data Type Fixes 

from pyspark.sql.types import IntegerType, DoubleType

df_clean = df_clean.withColumn("children", col("children").cast(DoubleType())) \
                   .withColumn("babies", col("babies").cast(IntegerType())) \
                   .withColumn("agent", col("agent").cast(IntegerType())) \
                   .withColumn("company", col("company").cast(IntegerType()))

df_clean.printSchema()


# 7 Feature Engineering / Transformations

from pyspark.sql.functions import expr, concat_ws

# Total stay length
df_clean = df_clean.withColumn(
    "total_nights",
    col("stays_in_weekend_nights") + col("stays_in_week_nights")
)

# Total guests
df_clean = df_clean.withColumn(
    "total_guests",
    col("adults") + col("children") + col("babies")
)

# Average revenue per booking
df_clean = df_clean.withColumn(
    "total_revenue",
    expr("adr * total_nights")
)

# Stay date (combine arrival year, month, day)
df_clean = df_clean.withColumn(
    "arrival_date",
    concat_ws("-", col("arrival_date_year"), col("arrival_date_month"), col("arrival_date_day_of_month"))
)

df_clean.select("hotel", "arrival_date", "total_nights", "total_revenue", "total_guests").show(5)

print(df_clean.show(5))
# 8 Save Curated Data (Parquet or Hive)

# Save as Parquet (preferred)
#df_clean.write.mode("overwrite").parquet("/tmp/DE011025/kavya/curated_hotel_bookings/")
