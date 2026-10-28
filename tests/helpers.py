"""Shared test helpers — importable from any test module."""
import os


class FakeUploadedFile:
    """Minimal stand-in for a Streamlit UploadedFile object."""

    def __init__(self, path: str):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._data = f.read()

    def getbuffer(self):
        return self._data
