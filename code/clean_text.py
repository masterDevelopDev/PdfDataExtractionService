from unidecode import unidecode


class TextCleaner:
    """
    A utility class for cleaning text.

    This class provides methods to perform basic cleaning operations on text data.
    It includes transliteration to ASCII and conversion to lowercase.

    Attributes:
        text (str): The text string to be cleaned.
    """
    def __init__(self, text):
        """
        The constructor for TextCleaner class.

        Parameters:
            text (str): The text string to be cleaned.
        """
        self.text = text

    def clean(self):
        """
        Clean the text by transliterating to ASCII and converting to lowercase.

        This is a wrapper method that calls the transliterate and lower_characters
        methods to perform text cleaning operations.

        Returns:
            str: The cleaned text.
        """
        return self.lower_characters(self.transliterate(self.text))

    @staticmethod
    def transliterate(text):
        """
        Convert the text to ASCII, ignoring non-ascii characters.

        Parameters:
            text (str): The text to transliterate.

        Returns:
            str: The ASCII transliterated text.
        """
        return unidecode(text)

    @staticmethod
    def lower_characters(text):
        return text.lower()
