from typing import Union
from pathlib import Path
from .base.document_processor import DocumentProcessor
from .pdf_processor import PDFProcessor
from .word_processor import WordProcessor
from .text_processor import TextProcessor
from .rtf_processor import RtfProcessor
from .code_processor import CodeProcessor
from .table_processor import TableProcessor

class ProcessorFactory:
    """Factory class for creating document processors"""

    @staticmethod
    def get_processor(file_path: Union[str, Path]) -> DocumentProcessor:
        """
        Get appropriate processor based on file extension

        Args:
        file_path: Path to the file

        Returns:
        DocumentProcessor: Appropriate processor for the file type

        Raises:
        ValueError: If file type is not supported
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if hasattr(file_path, 'name'):
            extension = Path(file_path.name).suffix.lower()
        else:
            extension = file_path.suffix.lower()

        if extension == '.pdf':
            return PDFProcessor()
        elif extension in ['.doc', '.docx']:
            return WordProcessor()
        elif extension == '.txt':
            return TextProcessor()
        elif extension == '.rtf':
            return RtfProcessor()
        elif TableProcessor.is_table_file(file_path):
            return TableProcessor()
        elif CodeProcessor.is_code_file(file_path):
            return CodeProcessor()

        raise ValueError("Please upload a supported file type: PDF, DOC, DOCX, TXT, RTF, Excel, CSV, ODS, JSON, Dockerfile, Markdown, YAML, or code file (e.g., .py, .js, .java, etc.)")