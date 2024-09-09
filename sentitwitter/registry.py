import torch
from google.cloud import storage

def save_model(model, bucket_name, model_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{model_path}.onnx")
    with blob.open("wb", ignore_flush=True) as f:
        f.write()