import logging
import os
import sys
from typing import Optional


# =============================================================================
# Color Definitions
# =============================================================================
class AnsiColors:
    """ANSI escape codes for terminal colors and formatting."""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


# =============================================================================
# Custom Formatter
# =============================================================================
class ColoredFormatter(logging.Formatter):
    """A custom log formatter that adds color to log levels and includes contextual information."""

    # Map log levels to their corresponding ANSI color codes
    LEVEL_COLORS = {
        logging.DEBUG: AnsiColors.BRIGHT_BLUE,
        logging.INFO: AnsiColors.BRIGHT_GREEN,
        logging.WARNING: AnsiColors.BRIGHT_YELLOW,
        logging.ERROR: AnsiColors.BRIGHT_RED,
        logging.CRITICAL: AnsiColors.RED,
    }

    def __init__(self, use_relative_path: bool = True):
        """
        Initialize the formatter.

        Args:
            use_relative_path (bool): If True, use relative path instead of absolute path
        """
        self.use_relative_path = use_relative_path
        self.project_root = self._find_project_root()
        super().__init__()

    def _find_project_root(self) -> Optional[str]:
        """Find the project root directory (looking for common markers)."""
        current_dir = os.getcwd()
        for _ in range(10):  # Limit the number of parent directories to check
            if any(
                os.path.exists(os.path.join(current_dir, marker))
                for marker in [".git", ".vscode", "pyproject.toml", "setup.py"]
            ):
                return current_dir
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # Reached the root directory
                break
            current_dir = parent_dir
        return None

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified log record as text with color and context.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message string.
        """
        # Get the color for the current log level
        color = self.LEVEL_COLORS.get(record.levelno, AnsiColors.WHITE)

        # Get the file path
        file_path = record.pathname
        if self.use_relative_path and self.project_root:
            try:
                file_path = os.path.relpath(file_path, self.project_root)
            except ValueError:
                # If relative path can't be created, use absolute path
                pass

        # Define the log format string. This includes:
        # - Timestamp (cyan)
        # - Logger name/namespace (magenta)
        # - Log level (color-coded)
        # - File path and line number (yellow) - VS Code recognizable format
        # - Function name (yellow)
        # - Log message
        log_format = (
            f"{AnsiColors.CYAN}[%(asctime)s]{AnsiColors.RESET} "
            f"{AnsiColors.MAGENTA}[%(name)s]{AnsiColors.RESET} "
            f"{color}[%(levelname)s]{AnsiColors.RESET} "
            f"{AnsiColors.YELLOW}{file_path}:%(lineno)d{AnsiColors.RESET} "
            f"{AnsiColors.BRIGHT_WHITE}%(funcName)s{AnsiColors.RESET} - "
            f"%(message)s"
        )

        # If an exception is associated with the record, ensure the traceback is included
        if record.exc_info:
            log_format += f"\n{AnsiColors.BRIGHT_RED}%(exc_text)s{AnsiColors.RESET}"

        # Create a Formatter instance with our format string and apply it
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# =============================================================================
# Logger Manager (Singleton Pattern)
# =============================================================================
class LoggerManager:
    """
    A singleton class to manage the application's logger instance.

    This ensures that only one logger is created and configured throughout the application.
    """

    _logger: Optional[logging.Logger] = None

    @classmethod
    def get_logger(
        cls, name: str = "NANODEPLOY", use_relative_path: bool = True
    ) -> logging.Logger:
        """
        Retrieve the singleton logger instance. If it doesn't exist, create and configure it.

        Args:
            name (str, optional): The name/namespace of the logger. Defaults to "NANODEPLOY".
            use_relative_path (bool, optional): If True, use relative paths in logs. Defaults to True.

        Returns:
            logging.Logger: The configured logger instance.
        """
        if cls._logger is not None:
            return cls._logger

        # Create a new logger with the specified name
        logger = logging.getLogger(name)
        # Set the logger's internal level to DEBUG to capture all levels of logs
        logger.setLevel(logging.DEBUG)
        # Prevent log messages from being propagated to the root logger (avoids duplication)
        logger.propagate = False

        # Check if handlers are already added to prevent duplicates in some environments
        if not logger.handlers:
            # Create a console handler to output logs to stdout
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)

            # Apply our custom colored formatter to the console handler
            formatter = ColoredFormatter(use_relative_path=use_relative_path)
            console_handler.setFormatter(formatter)

            # Add the handler to the logger
            logger.addHandler(console_handler)

            # Uncomment the following lines to also log to a file (optional)
            # file_handler = logging.FileHandler('nanodeploy.log', encoding='utf-8')
            # file_handler.setLevel(logging.INFO) # Log INFO and above to file
            # file_handler.setFormatter(logging.Formatter(
            #     '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
            #     datefmt='%Y-%m-%d %H:%M:%S'
            # ))
            # logger.addHandler(file_handler)

        # Store the configured logger instance
        cls._logger = logger
        return logger


# =============================================================================
# Module Exports
# =============================================================================
# Export a simple function for easy access to the logger
def get_logger(
    name: str = "NANODEPLOY", use_relative_path: bool = True
) -> logging.Logger:
    return LoggerManager.get_logger(name, use_relative_path)


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Obtain the logger instance
    # Set use_relative_path=False to use absolute paths
    logger = get_logger(use_relative_path=True)

    # Demonstrate different log levels
    logger.debug("This is a debug message, useful for development.")
    logger.info("This is an info message, indicating normal operation.")
    logger.warning("This is a warning message, indicating a potential issue.")
    logger.error("This is an error message, indicating a failure in an operation.")

    # Demonstrate logging an exception with a traceback
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("An unexpected exception occurred:")

    # Test with a different function
    def test_function():
        logger.info("This message is from test_function")

    test_function()
