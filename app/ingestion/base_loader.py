from abc import ABC, abstractmethod

from app.core.document import Document


class BaseLoader(ABC):

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load documents from a source.

        Args:
            source: File path or URL.

        Returns:
            List of Document objects.
        """
        pass