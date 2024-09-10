import logging

from sentitwitter.model import ClassModel
from sentitwitter.data import DataModule

import torch

import datetime


logger = logging.getLogger(__name__)


def convert_model():
    
    model_path = "./models/best-checkpoint.ckpt.ckpt"
    output_path = "./models"
        
    class_model = ClassModel.load_from_checkpoint(model_path)
    
    data_model = DataModule()
    
    data_model.prepare_data()
    data_model.setup()
    input_batch = next(iter(data_model.train_dataloader())) 
    input_sample = {
        "input_ids": input_batch["input_ids"],
        "attention_mask": input_batch["attention_mask"],
        "labels": input_batch["labels"]
    }
    
    current_stamp = datetime.datetime.now()
    
    model_name = f'model_{str(current_stamp)}'
    
    logger.info("Converting model to ONNX")
    torch.onnx.export(
        class_model,
        (input_sample["input_ids"], input_sample["attention_mask"]),
        f"{output_path}/{model_name}.onnx",
        export_params=True,
        opset_version=11,
        input_names=["input_ids", "attention_mask"],
        output_names=["output"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    logger.info(
        f"Model converted successfully. ONNX format model is at: {output_path}/{model_name}.onnx"
    )
    
    return f"{model_name}.onnx"


if __name__ == "__main__":
    convert_model()