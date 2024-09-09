import torch
from sentitwitter.model import ClassModel
from sentitwitter.data import DataModule
from scipy.special import softmax

import onnxruntime as ort

import numpy as np



class ClassONNXPredictor:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["Republican", "Democrat"]


    def predict(self, text):
        inference_sample = {"text": text}
        processed = self.processor.tokenize_data(inference_sample)
                
        ort_inputs = {
            "input_ids": np.expand_dims(processed["input_ids"], axis=0),
            "attention_mask": np.expand_dims(processed["attention_mask"], axis=0),
        }
        
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        scores = softmax(ort_outs[0])[0]
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"labels": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "Right-wing extremists are fanning the flames of hate and picking on trans children in states across the country. It’s sickening."
    predictor = ClassONNXPredictor("./models/model.onnx")
    print(predictor.predict(sentence))
