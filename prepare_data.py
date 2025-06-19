import boto3
import urllib.request
import zipfile
import os

def download_movielens_data():
    
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    local_zip = "ml-latest-small.zip"
    
    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(url, local_zip)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    dataset_path = "./ml-latest-small"   
    print("Dataset ready at: ./ml-latest-small/")
    return dataset_path

def upload_movielens_data(data_path,bucket_name="recommender-movielens-slv", prefix="movielens"):
    s3=boto3.client('s3')
    files_to_upload = ['ratings.csv', 'movies.csv']
    for filename in files_to_upload:
        local_path = os.path.join(data_path, filename)
        s3_key = f"{prefix}/{filename}"
        try:
            print(f"Uploading {filename} to s3://{bucket_name}/{s3_key} ...")
            s3.upload_file(local_path, bucket_name, s3_key)
            
        except Exception as e:
            print(f"Failed to upload {filename}: {e}")

    
if __name__ == "__main__":
    data_path = download_movielens_data()
    upload_movielens_data(data_path)



