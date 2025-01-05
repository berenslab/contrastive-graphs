class IndentedWriter:
    """
    A context manager that adds indentation to written output.
    The indent level can be changed using push() and pop() methods.

    Usage:
        writer = IndentedWriter(sys.stdout)
        writer.write("No indent")
        with writer.indent():
            writer.write("First level")
            with writer.indent():
                writer.write("Second level")
        writer.write("Back to no indent")
    """

    def __init__(self, file, indent_str="    "):
        self.file = file
        self.indent_str = indent_str
        self._indent_level = 0

    def write(self, text):
        """Write text with proper indentation."""
        # Split the text into lines
        lines = text.split("\n")

        # Get the current indent string based on level
        indent = self.indent_str * self._indent_level

        # Write each line with proper indentation
        for i, line in enumerate(lines):
            # Only indent non-empty lines
            if line:
                self.file.write(indent + line)
            if i < len(lines) - 1:  # Don't write newline after last line
                self.file.write("\n")

    def writeln(self, text):
        self.write(text)
        self.write("\n")

    def push(self):
        """Increase the indent level."""
        self._indent_level += 1
        return self

    def pop(self):
        """Decrease the indent level."""
        if self._indent_level > 0:
            self._indent_level -= 1
        return self

    def indent(self):
        """Context manager for temporary indent level change."""
        return _IndentContext(self)


class _IndentContext:
    """Helper class to manage indent levels in a context."""

    def __init__(self, writer):
        self.writer = writer

    def __enter__(self):
        self.writer.push()
        return self.writer

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.pop()
