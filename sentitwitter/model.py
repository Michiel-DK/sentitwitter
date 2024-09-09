import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import AutoModel,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import torchmetrics

import wandb



class ClassModel(pl.LightningModule):
    def __init__(self, model_name="distilbert/distilbert-base-uncased", lr=1e-2):
        super(ClassModel, self).__init__()
        self.save_hyperparameters()

        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2
        self.task = 'binary'
        
        self.training_step_outputs = []
        self.training_step_targets = []

        self.val_step_outputs = []
        self.val_step_targets = []
        
        #metrics
        self.train_accuracy_metric = torchmetrics.Accuracy(task = self.task )
        self.val_accuracy_metric = torchmetrics.Accuracy(task = self.task )
        self.f1_metric = torchmetrics.F1Score(task=self.task ,num_classes=self.num_classes)
        self.precision_macro_metric = torchmetrics.Precision(task = self.task ,
            average="macro", num_classes=self.num_classes
        )
        self.recall_macro_metric = torchmetrics.Recall(task = self.task ,
            average="macro", num_classes=self.num_classes
        )
        self.precision_micro_metric = torchmetrics.Precision(task = self.task, average="micro")
        self.recall_micro_metric = torchmetrics.Recall(task = self.task , average="micro")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def training_step(self, batch, batch_idx):
        
        
        outputs = self.forward(
            batch["input_ids"], batch["attention_mask"], labels=batch["labels"]
        )
        
        preds = torch.argmax(outputs.logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["labels"])
        
        self.training_step_outputs.append(preds)
        self.training_step_targets.append(batch["labels"])
        
        self.log("train/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)
        
        return outputs.loss
    
    def on_train_epoch_end(self):
        
        self.training_step_outputs.clear()
        self.training_step_targets.clear()

    def validation_step(self, batch, batch_idx):
        
        labels = batch["labels"]
        
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        preds = torch.argmax(outputs.logits, 1)
        
        #import ipdb;ipdb.set_trace()
        
        self.val_step_outputs.append(preds)
        self.val_step_targets.append(labels)
                
        # Metrics
        valid_acc = self.val_accuracy_metric(preds, labels)
        precision_macro = self.precision_macro_metric(preds, labels)
        recall_macro = self.recall_macro_metric(preds, labels)
        precision_micro = self.precision_micro_metric(preds, labels)
        recall_micro = self.recall_micro_metric(preds, labels)
        f1 = self.f1_metric(preds, labels)

        # Logging metrics
        self.log("valid/loss", outputs.loss, prog_bar=True, on_epoch=True)
        self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True)
        self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True)
        self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True)
        self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True)
        self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
        return {"labels": labels, "logits": outputs.logits}

    def on_validation_epoch_end(self):
        
        #labels = torch.cat([x["labels"] for x in self.validation_step_outputs[-1]])
        #logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        #import ipdb;ipdb.set_trace()
        preds = self.val_step_outputs[0]
        labels = self.val_step_targets[0]
                        
        self.val_step_outputs.clear()
        self.val_step_targets.clear()

        ## There are multiple ways to track the metrics
        # # 1. Confusion matrix plotting using inbuilt W&B method
        # self.logger.experiment.log(
        #     {
        #         "conf": wandb.plot.confusion_matrix(
        #             probs=logits.numpy(), y_true=labels.numpy()
        #         )
        #     }
        # )
        
        # 2. Confusion Matrix plotting using scikit-learn method
        wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])