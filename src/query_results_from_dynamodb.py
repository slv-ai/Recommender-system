import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('movie-recommendations')

# Get recommendations for a specific user
response = table.query(
    KeyConditionExpression='userId = :userId',
    ExpressionAttributeValues={':userId': 1}
)

print("Recommendations for User 1:")
for item in response['Items']:
    print(f"Movie {item['movieId']}: {item['predicted_rating']:.2f}")