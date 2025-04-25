"""
Beautiful, production-standard logging configuration.

Features:
- Colorized console output for better readability
- Log rotation with date-based naming
- Centralized logs in data/logs directory
- Contextual information in log entries
"""
import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional

# Try to import colorama for colored console output
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)  # Initialize colorama with autoreset
    COLORAMA_AVAILABLE = True
except ImportError:
    # Create dummy color classes if colorama is not available
    class DummyColor:
        def __getattr__(self, name):
            return ""

    Fore = DummyColor()
    Back = DummyColor()
    Style = DummyColor()
    COLORAMA_AVAILABLE = False
    print("Note: colorama is not installed. Install it for colorized output: pip install colorama")

# Environment-based configuration
ENV = os.getenv('APP_ENV', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Find the project root directory (where data/logs should be)
try:
    # Start from the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try multiple approaches to find the project root
    # Approach 1: Go up from src/shared to src, then to project root
    if os.path.basename(current_dir) == 'shared':
        src_dir = os.path.dirname(current_dir)
        if os.path.basename(src_dir) == 'src':
            project_root = os.path.dirname(src_dir)
        else:
            project_root = os.path.dirname(current_dir)  # Fallback
    else:
        # Approach 2: Look for common project directories
        project_root = current_dir
        while project_root and not os.path.exists(os.path.join(project_root, 'src')) and os.path.dirname(project_root) != project_root:
            project_root = os.path.dirname(project_root)

    # Create the data/logs directory at the project root
    LOG_DIR = os.path.join(project_root, 'data', 'logs')
    os.makedirs(LOG_DIR, exist_ok=True)

    # Check if directory is writable
    test_file = os.path.join(LOG_DIR, 'test_write.tmp')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)

    print(f"Log directory created and is writable: {LOG_DIR}")
except Exception as e:
    print(f"WARNING: Could not create or write to log directory: {str(e)}")
    # Fallback to a directory we know should be writable
    LOG_DIR = os.path.abspath(os.path.expanduser('~/data_analysis_tool_logs'))
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Using fallback log directory: {LOG_DIR}")

# Single log file path
LOG_FILE = os.path.join(LOG_DIR, 'logs.log')
print(f"Logs will be written to: {LOG_FILE}")

# Log format for console and file
CONSOLE_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
FILE_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(process)d:%(thread)d] - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

class ColoredFormatter(logging.Formatter):
    """
    Simple formatter for beautiful colorized console output.
    """

    # Define color schemes for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Save the original levelname
        levelname = record.levelname

        # Apply color to the levelname based on level
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"

        # Apply color to the message based on level
        if levelname in self.COLORS:
            record.msg = f"{self.COLORS[levelname]}{record.msg}{Style.RESET_ALL}"

        # Format the record
        return super().format(record)

# We don't need a JSON formatter for this simple implementation

def configure_logging(module_name: str):
    """
    Configure logging for a module with beautiful, production-standard settings.

    Args:
        module_name: Name of the module (e.g., 'frontend.pages.analysis')

    Returns:
        Logger instance
    """
    logger = logging.getLogger(module_name)

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    console_formatter = ColoredFormatter(CONSOLE_FORMAT, DATE_FORMAT)
    file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # File handler with daily rotation
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        # Set up the file handler
        file_handler = TimedRotatingFileHandler(
            LOG_FILE,
            when='midnight',
            interval=1,
            backupCount=30  # Keep logs for 30 days
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)

        # Add file handler to logger
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"WARNING: Could not create log file handler: {str(e)}")

    # Always add console handler
    logger.addHandler(console_handler)

    return logger

class ContextLogger:
    """
    Simple context-aware logger that adds context to log messages.
    """

    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.context = {}

    def add_context(self, **kwargs):
        """Add context data that will be included in all subsequent log messages."""
        self.context.update(kwargs)
        return self

    def clear_context(self):
        """Clear all context data."""
        self.context = {}
        return self

    def _log_with_context(self, level, msg, *args, **kwargs):
        """Log a message with the current context."""
        if args:
            msg = msg % args

        # Add context to the message if available
        if self.context:
            context_str = ' '.join(f"[{k}={v}]" for k, v in self.context.items())
            msg = f"{msg} {context_str}"

        # Call the appropriate logging method
        getattr(self.logger, level)(msg, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log_with_context('debug', msg, *args, **kwargs)
        return self

    def info(self, msg, *args, **kwargs):
        self._log_with_context('info', msg, *args, **kwargs)
        return self

    def warning(self, msg, *args, **kwargs):
        self._log_with_context('warning', msg, *args, **kwargs)
        return self

    def error(self, msg, *args, **kwargs):
        self._log_with_context('error', msg, *args, **kwargs)
        return self

    def critical(self, msg, *args, **kwargs):
        self._log_with_context('critical', msg, *args, **kwargs)
        return self

    def exception(self, msg, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log_with_context('error', msg, *args, **kwargs)
        return self

def get_logger(name: str):
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    # Check if we already have a logger with this name
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Return existing logger

    # Configure a new logger
    return configure_logging(name)

def get_context_logger(name: str):
    """Get a context logger with the specified name."""
    return ContextLogger(name)

def test_logging():
    """Test the beautiful, production-standard logging configuration."""
    # Create a colorful header
    header = f"{Back.BLUE}{Fore.WHITE}{Style.BRIGHT} LOGGING TEST {Style.RESET_ALL}"
    separator = f"{Fore.BLUE}{'='*80}{Style.RESET_ALL}"

    print("\n" + separator)
    print(header)
    print(separator)

    # Print information about the log file
    print(f"\n{Fore.CYAN}Log file path:{Style.RESET_ALL} {LOG_FILE}")

    # Write test log messages
    logger = get_logger("test_logging")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Test context logging
    context_logger = get_context_logger("test_context_logging")
    context_logger.add_context(user="test_user", action="login").info("User logged in")

    # Try to read and display the last few lines of the log file
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                if lines:
                    print(f"\n{Fore.GREEN}Last {min(5, len(lines))} lines from the log file:{Style.RESET_ALL}")
                    for line in lines[-5:]:
                        print(f"  {line.strip()}")
        else:
            print(f"\n{Fore.YELLOW}Log file does not exist yet.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Could not read log file: {str(e)}{Style.RESET_ALL}")

    # Create a colorful footer
    footer = f"{Back.GREEN}{Fore.BLACK}{Style.BRIGHT} TEST COMPLETE {Style.RESET_ALL}"
    print("\n" + separator)
    print(footer)
    print(separator)

# Uncomment to test logging when this module is imported
# test_logging()
