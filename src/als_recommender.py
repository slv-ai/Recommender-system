import sys
import boto3
import json
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col,explode, lit
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class MovieRecommendationEngine:
    def __init__(self):
        self.spark = (SparkSession.builder
            .appName("MovieRecommendationEngine")
            .config("fs.s3a.connection.timeout", "300000")
            .config("fs.s3a.connection.establish.timeout", "300000")
            .config("fs.s3a.attempts.maximum", "10")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
            .getOrCreate()
        )       
        self.dynamodb=boto3.resource('dynamodb',region_name='us-east-1')
        self.recommendations_table=self.dynamodb.Table('movie-recommendations')
        self.metadata_table=self.dynamodb.Table("model-metadata")
    
    def load_movielens_data(self,data_path):
        logger.info("loading movielens data")
        # Load ratings data
        ratings = self.spark.read.csv(f"{data_path}/ratings.csv",header=True, inferSchema=True)
        
        # Load movies data for metadata
        movies = self.spark.read.csv(f"{data_path}/movies.csv",header=True, inferSchema=True)
        logger.info(f"loaded {ratings.count()} ratings and {movies.count()} movies")

        return ratings,movies

    def prepare_data(self,ratings):
        logger.info("prepare data for training")
        #split data into training and test
        (training,test)=ratings.randomSplit([0.8,0.2],seed=42)

        logger.info(f"Training set: {training.count()} ratings")
        logger.info(f"Test set: {test.count()} ratings")
        
        return training, test

    def train_als_model(self, training_data):
        logger.info("Training ALS model...")
        
       
        als = ALS(maxIter=10, 
                 regParam=0.1, 
                 rank=10,
                 userCol="userId", 
                 itemCol="movieId", 
                 ratingCol="rating",
                 coldStartStrategy="drop",
                 seed=42)
        
        # Train model
        model = als.fit(training_data)
        
        logger.info("ALS model training completed")
        return model

    def save_model_to_s3(self,model):
        S3_BUCKET = "recommender-movielens-slv"  
        MODEL_VERSION=datetime.now().strftime("%Y%m%d-%H%M")
        model_path = f"s3a://{S3_BUCKET}/movielens/models/{MODEL_VERSION}"   
        model.write.overwrite().save(model_path)
        logger.info(f"Model saved to {model_path}")
        return model_path
       
    def evaluate_model(self, model, test_data):
       
        logger.info("Evaluating model...")
        
        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(metricName="rmse", 
                                      labelCol="rating",
                                      predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        
        logger.info(f"Root Mean Square Error (RMSE): {rmse}")
        return rmse

    def generate_recommendations(self, model, num_recommendations=10):
        logger.info("Generating recommendations...")
        
        # Generate top movie recommendations for all users
        user_recs = model.recommendForAllUsers(num_recommendations)
        
        # Flatten the recommendations
        user_recs_flat = user_recs.select(
            col("userId"),
            explode(col("recommendations")).alias("recommendation")
        ).select(
            col("userId"),
            col("recommendation.movieId").alias("movieId"),
            col("recommendation.rating").alias("predicted_rating")
        )
        
        logger.info(f"Generated recommendations for {user_recs.count()} users")
        return user_recs_flat

    def save_to_dynamodb(self, recommendations_df, model_metadata):
        logger.info("Saving recommendations to DynamoDB...")
        
        # Convert Spark DataFrame to list of dictionaries
        recommendations_list = recommendations_df.collect()
        
        # Batch write to DynamoDB
        with self.recommendations_table.batch_writer() as batch:
            for row in recommendations_list:
                batch.put_item(
                    Item={
                        'userId': int(row['userId']),
                        'movieId': int(row['movieId']),
                        'predicted_rating': float(row['predicted_rating']),
                        'timestamp': datetime.now().isoformat()
                    }
                )
        
        # Save model metadata
        self.metadata_table.put_item(
            Item={
                'modelId': model_metadata['modelId'],
                'rmse': model_metadata['rmse'],
                'training_date': model_metadata['training_date'],
                'num_recommendations': model_metadata['num_recommendations']
            }
        )
        
        logger.info("Successfully saved to DynamoDB")

     
    def run_pipeline(self, data_path):
        
        try:
            # Load data
            ratings, movies = self.load_movielens_data(data_path)
            
            # Prepare data
            training, test = self.prepare_data(ratings)
            
            # Train model
            model = self.train_als_model(training)

            #save model to s3
            self.save_model_to_s3(model)
            
            # Evaluate model
            rmse = self.evaluate_model(model, test)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(model)
            
            # Prepare metadata
            model_metadata = {
                'modelId': f"als_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'rmse': float(rmse),
                'training_date': datetime.now().isoformat(),
                'num_recommendations': recommendations.count()
            }
            
            # Save to DynamoDB
            self.save_to_dynamodb(recommendations, model_metadata)
            
            logger.info("Pipeline completed successfully!")
            return {"status": "success", "rmse": rmse}

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {"status": "error", "message": str(e)}
        
        finally:
            if hasattr(self, 'spark'):
                self.spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 recommendation_engine.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    engine = MovieRecommendationEngine()
    result = engine.run_pipeline(data_path)
    print(json.dumps(result))