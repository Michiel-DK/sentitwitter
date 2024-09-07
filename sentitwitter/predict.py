import torch
from sentitwitter.model import ClassModel
from sentitwitter.data import DataModule


class ColaPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = ClassModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()
        self.processor = DataModule()
        self.softmax = torch.nn.Softmax(dim=0)
        self.lables = ["Republican", "Democrat"]

    def predict(self, text):
        inference_sample = {"text": text}
        processed = self.processor.tokenize_data(inference_sample)
        logits = self.model(
            torch.tensor([processed["input_ids"]]).to('mps'),
            torch.tensor([processed["attention_mask"]]).to('mps'),
        )
        scores = self.softmax(logits[0]).tolist()
        predictions = []
        for score, label in zip(scores, self.lables):
            predictions.append({"labels": label, "score": score})
        return predictions


if __name__ == "__main__":
    sentence = "Right-wing extremists are fanning the flames of hate and picking on trans children in states across the country. It’s sickening."
    predictor = ColaPredictor("./models/epoch=0-step=499.ckpt")
    print(predictor.predict(sentence))
