import csv

from naive_bayes import NaiveBayesClassifier
from evaluator import ModelEvaluator


def load_dataset(file_path):
    """
    Load emails and labels from a CSV file.
    """
    emails = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labels.append(row["label"].strip())
            emails.append(row["text"].strip())

    return emails, labels


def print_confusion_matrix(tp, tn, fp, fn):
    """
    Print the confusion matrix in a clean format.
    """
    print("\nConfusion Matrix:")
    print("-----------------")
    print(f"True Positives  (Spam -> Spam): {tp}")
    print(f"True Negatives  (Ham  -> Ham):  {tn}")
    print(f"False Positives (Ham  -> Spam): {fp}")
    print(f"False Negatives (Spam -> Ham):  {fn}")


def main():
    print("Spam Email Classifier Using Naive Bayes")
    print("=======================================")

    train_file = "data/train_emails.csv"
    test_file = "data/test_emails.csv"

    train_emails, train_labels = load_dataset(train_file)
    test_emails, test_labels = load_dataset(test_file)

    classifier = NaiveBayesClassifier()
    evaluator = ModelEvaluator()

    classifier.train(train_emails, train_labels)

    predicted_labels = []
    print("\nTest Predictions:")
    print("-----------------")
    for email, actual_label in zip(test_emails, test_labels):
        predicted_label, spam_score, ham_score = classifier.predict_with_scores(email)
        predicted_labels.append(predicted_label)

        print(f"Email: {email}")
        print(f"Actual Label:    {actual_label}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Spam Score: {spam_score:.4f}")
        print(f"Ham Score:  {ham_score:.4f}")
        print()

    tp, tn, fp, fn = evaluator.confusion_matrix(test_labels, predicted_labels)
    accuracy = evaluator.accuracy(test_labels, predicted_labels)
    precision = evaluator.precision(test_labels, predicted_labels)
    recall = evaluator.recall(test_labels, predicted_labels)
    f1 = evaluator.f1_score(test_labels, predicted_labels)

    print_confusion_matrix(tp, tn, fp, fn)

    print("\nEvaluation Metrics:")
    print("-------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nCustom Email Classification")
    print("---------------------------")
    user_email = input("Enter an email message to classify: ").strip()

    if user_email:
        predicted_label, spam_score, ham_score = classifier.predict_with_scores(user_email)
        print("\nResult:")
        print(f"Predicted Label: {predicted_label}")
        print(f"Spam Score: {spam_score:.4f}")
        print(f"Ham Score:  {ham_score:.4f}")
    else:
        print("No email entered. Program finished.")


if __name__ == "__main__":
    main()