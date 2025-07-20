from pathlib import Path
from typing import ClassVar

from markitdown import MarkItDown


class FileReader:
    extensions: ClassVar[list[str]] = [
        ".astro",
        ".c",
        ".cpp",
        ".css",
        ".csv",
        ".docx",
        ".go",
        ".h",
        ".hpp",
        ".html",
        ".java",
        ".js",
        ".json",
        ".kt",
        ".md",
        ".mdx",
        ".mjs",
        ".mp3",
        ".pdf",
        ".php",
        ".pptx",
        ".py",
        ".rb",
        ".rs",
        ".svelte",
        ".swift",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".wav",
        ".xml",
        ".xlsx",
        ".yaml",
        ".yml",
    ]

    @staticmethod
    def parse_file(path: Path) -> str:
        try:
            reader = MarkItDown()
            return reader.convert(path).text_content
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle PDF color space errors
            if "cannot set non-stroke color" in error_msg and path.suffix.lower() == ".pdf":
                raise ValueError(
                    f"PDF file '{path.name}' contains unsupported color format. "
                    f"This PDF uses a color space that is not supported (possibly duotone or Lab colors). "
                    f"Try converting the PDF to standard RGB format or extracting the text content manually."
                )
            
            # Handle other errors
            raise ValueError(f"Failed to parse file '{path.name}': {str(e)}")
