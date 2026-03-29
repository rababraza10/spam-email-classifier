import math
from collections import defaultdict

from preprocessor import TextPreprocessor


class NaiveBayesClassifier:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

        self.vocabulary = set()
        self.class_counts = {"spam": 0, "ham": 0}
        self.word_counts = {
            "spam": defaultdict(int),
            "ham": defaultdict(int)
        }
        self.total_words = {"spam": 0, "ham": 0}
        self.priors = {"spam": 0.0, "ham": 0.0}
        self.is_trained = False

    def train(self, emails, labels):
        """
        Train the Naive Bayes classifier using the training data.
        """
        total_emails = len(emails)

        for email, label in zip(emails, labels):
            self.class_counts[label] += 1

            words = self.preprocessor.tokenize(email)

            for word in words:
                self.vocabulary.add(word)
                self.word_counts[label][word] += 1
                self.total_words[label] += 1

        self.priors["spam"] = self.class_counts["spam"] / total_emails
        self.priors["ham"] = self.class_counts["ham"] / total_emails

        self.is_trained = True

    def word_likelihood(self, word, label):
        """
        Calculate P(word | label) using Laplace smoothing.
        """
        word_count = self.word_counts[label][word]
        total_label_words = self.total_words[label]
        vocabulary_size = len(self.vocabulary)

        likelihood = (word_count + 1) / (total_label_words + vocabulary_size)
        return likelihood

    def predict(self, email):
        """
        Predict whether a new email is spam or ham.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        words = self.preprocessor.tokenize(email)

        spam_score = math.log(self.priors["spam"])
        ham_score = math.log(self.priors["ham"])

        for word in words:
            spam_score += math.log(self.word_likelihood(word, "spam"))
            ham_score += math.log(self.word_likelihood(word, "ham"))

        if spam_score > ham_score:
            return "spam"
        return "ham"

    def predict_with_scores(self, email):
        """
        Predict the label and also return spam and ham scores.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")

        words = self.preprocessor.tokenize(email)

        spam_score = math.log(self.priors["spam"])
        ham_score = math.log(self.priors["ham"])

        for word in words:
            spam_score += math.log(self.word_likelihood(word, "spam"))
            ham_score += math.log(self.word_likelihood(word, "ham"))

        predicted_label = "spam" if spam_score > ham_score else "ham"

        return predicted_label, spam_score, ham_score