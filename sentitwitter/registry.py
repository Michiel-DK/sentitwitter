from google.cloud import storage

def save_model(bucket_name, model_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"models/{model_name}")
    blob.upload_from_filename(f"models/{model_name}", timeout=300)