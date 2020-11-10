import torch
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall, Fbeta
from pytorch_lightning.metrics.utils import METRIC_EPS


def accuracy_score(labels, predictions):
    predictions = torch.Tensor(predictions)
    accuracy = Accuracy(threshold=0.5)
    return accuracy(predictions, labels).item()


def precision_score(labels, predictions):
    predictions = torch.Tensor(predictions)
    precision = Precision(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    return precision(predictions, labels).item()


def class_wise_precision_scores(labels, predictions):
    predictions = torch.Tensor(predictions)
    precision = Precision(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    precision(predictions, labels)
    class_wise_scores = precision.true_positives.float() / (precision.predicted_positives + METRIC_EPS)
    return class_wise_scores.tolist()


def recall_score(labels, predictions):
    predictions = torch.Tensor(predictions)
    recall = Recall(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    return recall(predictions, labels).item()


def class_wise_recall_scores(labels, predictions):
    predictions = torch.Tensor(predictions)
    recall = Recall(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    recall(predictions, labels)
    class_wise_scores = recall.true_positives.float() / (recall.actual_positives + METRIC_EPS)
    return class_wise_scores.tolist()


def f1_score(labels, predictions):
    predictions = torch.Tensor(predictions)
    f_beta = Fbeta(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    return f_beta(predictions, labels).item()


def class_wise_f1_scores(labels, predictions):
    predictions = torch.Tensor(predictions)
    f_beta = Fbeta(num_classes=len(labels[0]), average='macro', multilabel=True, threshold=0.5)
    f_beta(predictions, labels)
    precision = f_beta.true_positives.float() / (f_beta.predicted_positives + METRIC_EPS)
    recall = f_beta.true_positives.float() / (f_beta.actual_positives + METRIC_EPS)
    class_wise_scores = 2*(precision * recall) / (precision + recall + METRIC_EPS)
    return class_wise_scores.tolist()


if __name__ == "__main__":

    truth = torch.Tensor([[1, 1, 0, 0, 1], [0, 0, 1, 1, 1], [1, 0, 0, 0, 1]])
    pred = torch.Tensor([[0.2, 0.7, 0.6, 0.8, 0.7], [1, 0.2, 0.3, 0.76, 0.98], [0.56, 0.67, 0.78, 0.23, 0.87]])

    print(accuracy_score(pred, truth))
    print(precision_score(pred, truth))
    print(recall_score(pred, truth))
    print(class_wise_precision_scores(pred, truth))
    print(class_wise_recall_scores(pred, truth))
    print(class_wise_f1_scores(pred, truth))
