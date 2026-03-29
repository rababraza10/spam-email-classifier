import string


class TextPreprocessor:
    def __init__(self):
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean_text(self, text):
        """
        Convert text to lowercase and remove punctuation.
        """
        text = text.lower()
        text = text.translate(self.punctuation_table)
        return text

    def tokenize(self, text):
        """
        Clean the text and split it into individual words.
        """
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        return tokens