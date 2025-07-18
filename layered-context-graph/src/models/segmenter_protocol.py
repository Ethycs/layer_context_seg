from abc import ABC, abstractmethod
from typing import List

class Segmenter(ABC):
    """
    An abstract base class (interface) for models that can segment text based
    on a natural language rule. This ensures that any class claiming to be a
    Segmenter implements the required `segment` method.
    """
    
    @abstractmethod
    def segment(self, rule: str, text_to_segment: str) -> List[str]:
        """
        Segments the given text according to the provided rule.

        Args:
            rule: A natural language instruction describing how to segment the text.
            text_to_segment: The text to be segmented.

        Returns:
            A list of strings, where each string is a segment of the original text.
        """
        raise NotImplementedError("Any class that inherits from Segmenter must implement the 'segment' method.")
