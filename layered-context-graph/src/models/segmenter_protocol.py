from typing import Protocol, List

class Segmenter(Protocol):
    """
    A protocol for models that can segment text based on a natural language rule.
    """
    def segment(self, rule: str, text_to_segment: str) -> List[str]:
        """
        Segments the given text according to the provided rule.

        Args:
            rule: A natural language instruction describing how to segment the text.
            text_to_segment: The text to be segmented.

        Returns:
            A list of strings, where each string is a segment of the original text.
        """
        ...
