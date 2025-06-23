# Recommender-system
An end-to-end movie recommendation system built with PySpark ML ALS, AWS DynamoDB and Step Functions. This project demonstrates collaborative filtering using the MovieLens dataset with a fully automated ML pipeline.


### Create recommendations table
aws dynamodb create-table \
    --table-name movie-recommendations \
    --attribute-definitions \
        AttributeName=userId,AttributeType=N \
        AttributeName=movieId,AttributeType=N \
    --key-schema \
        AttributeName=userId,KeyType=HASH \
        AttributeName=movieId,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST

### Create model metadata table
aws dynamodb create-table \
    --table-name model-metadata \
    --attribute-definitions \
        AttributeName=modelId,AttributeType=S \
    --key-schema \
        AttributeName=modelId,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

### verify tables are created
aws dynamodb list-tables
###to inspect schema
aws dynamodb describe-table --table-name movie-recommendations

### Run the script:
python3 upload_movielens_to_s3.py

### Run the recommendation engine
python3 als_recommender.py s3a://recommender-movielens-slv/movielens

spark-submit \
  --packages org.apache.hadoop:hadoop-aws:3.3.1 \
  als_recommender.py s3a://recommender-movielens-slv/movielens






