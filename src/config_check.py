from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("MyApp")
    .config("spark.network.timeout", "60000")
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.5,com.amazonaws:aws-java-sdk-bundle:1.12.534")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
    .getOrCreate()
)

print("Starting read from S3...")
df = spark.read.option("header", True).csv("s3a://recommender-movielens-slv/movielens/ratings.csv")
print("Read schema:", df.schema.simpleString())
df.show(5)

spark.stop()





