import boto3
import urllib.request
import zipfile
import os

def download_movielens_data():
    
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    
    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(url, "ml-latest-small.zip")
    
    print("Extracting dataset...")
    with zipfile.ZipFile("ml-latest-small.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("Dataset ready at: ./ml-latest-small/")
    return "./ml-latest-small"

if __name__ == "__main__":
    data_path = download_movielens_data()
    print(f"Data available at: {data_path}")