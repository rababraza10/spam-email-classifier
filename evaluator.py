class ModelEvaluator:
    def __init__(self):
        pass

    def confusion_matrix(self, actual_labels, predicted_labels):
        """
        Return TP, TN, FP, FN for spam classification.
        Positive class = spam
        Negative class = ham
        """
        tp = tn = fp = fn = 0

        for actual, predicted in zip(actual_labels, predicted_labels):
            if actual == "spam" and predicted == "spam":
                tp += 1
            elif actual == "ham" and predicted == "ham":
                tn += 1
            elif actual == "ham" and predicted == "spam":
                fp += 1
            elif actual == "spam" and predicted == "ham":
                fn += 1

        return tp, tn, fp, fn

    def accuracy(self, actual_labels, predicted_labels):
        tp, tn, fp, fn = self.confusion_matrix(actual_labels, predicted_labels)
        total = tp + tn + fp + fn
        if total == 0:
            return 0.0
        return (tp + tn) / total

    def precision(self, actual_labels, predicted_labels):
        tp, tn, fp, fn = self.confusion_matrix(actual_labels, predicted_labels)
        denominator = tp + fp
        if denominator == 0:
            return 0.0
        return tp / denominator

    def recall(self, actual_labels, predicted_labels):
        tp, tn, fp, fn = self.confusion_matrix(actual_labels, predicted_labels)
        denominator = tp + fn
        if denominator == 0:
            return 0.0
        return tp / denominator

    def f1_score(self, actual_labels, predicted_labels):
        precision_value = self.precision(actual_labels, predicted_labels)
        recall_value = self.recall(actual_labels, predicted_labels)

        denominator = precision_value + recall_value
        if denominator == 0:
            return 0.0

        return 2 * (precision_value * recall_value) / denominator