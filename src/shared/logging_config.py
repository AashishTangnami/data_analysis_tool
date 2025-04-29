"""
Simple, production-grade logging configuration.

Features:
- Logs file_id, file_type, file_size
- Logs API request and endpoint
- Logs errors and successes
- Colorized console output
- Structured JSON logging
- Path masking for security
"""
import os
import sys
import json
import uuid
import socket
import logging
import traceback
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Any, Optional, Union, List

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

# List of fields that should be masked in logs
SENSITIVE_FIELDS = [
    "password", "token", "secret", "key", "auth", "credential", "private",
    "ssn", "social", "credit", "card", "cvv", "access_token", "refresh_token"
]

# Thread-local storage for context variables
_thread_local = threading.local()

def get_correlation_id():
    """Get the correlation ID for the current thread."""
    if not hasattr(_thread_local, 'correlation_id'):
        _thread_local.correlation_id = str(uuid.uuid4())
    return _thread_local.correlation_id

def set_correlation_id(correlation_id):
    """Set the correlation ID for the current thread."""
    _thread_local.correlation_id = correlation_id

def set_request_context(endpoint=None, method=None):
    """
    Set API request context for logging.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
    """
    if not hasattr(_thread_local, 'request_context'):
        _thread_local.request_context = {}

    if endpoint:
        _thread_local.request_context['endpoint'] = endpoint
    if method:
        _thread_local.request_context['method'] = method

def get_request_context():
    """Get the request context for the current thread."""
    if not hasattr(_thread_local, 'request_context'):
        _thread_local.request_context = {}
    return _thread_local.request_context

def set_file_context(file_id=None, file_type=None, file_size=None, file_name=None):
    """
    Set file context for logging.

    Args:
        file_id: Unique identifier for the file
        file_type: MIME type or format of the file
        file_size: Size of the file in bytes
        file_name: Name of the file
    """
    if not hasattr(_thread_local, 'file_context'):
        _thread_local.file_context = {}

    if file_id:
        _thread_local.file_context['file_id'] = file_id
    if file_type:
        _thread_local.file_context['file_type'] = file_type
    if file_size is not None:
        # Convert to MB and round to 2 decimal places if size is in bytes
        if file_size > 1024:
            file_size_mb = round(file_size / (1024 * 1024), 2)
            _thread_local.file_context['file_size_mb'] = file_size_mb
        else:
            _thread_local.file_context['file_size'] = file_size
    if file_name:
        # Only store the filename, not the path
        _thread_local.file_context['file_name'] = os.path.basename(file_name)

def get_file_context():
    """Get the file context for the current thread."""
    if not hasattr(_thread_local, 'file_context'):
        _thread_local.file_context = {}
    return _thread_local.file_context

def clear_context():
    """Clear all context variables for the current thread."""
    if hasattr(_thread_local, 'correlation_id'):
        delattr(_thread_local, 'correlation_id')
    if hasattr(_thread_local, 'request_context'):
        delattr(_thread_local, 'request_context')
    if hasattr(_thread_local, 'file_context'):
        delattr(_thread_local, 'file_context')

def mask_sensitive_data(data: Any) -> Any:
    """
    Recursively mask sensitive data in logs.

    Args:
        data: Data to mask

    Returns:
        Masked data
    """
    if isinstance(data, dict):
        return {
            k: "********" if any(sensitive in k.lower() for sensitive in SENSITIVE_FIELDS)
            else mask_sensitive_data(v)
            for k, v in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return [mask_sensitive_data(item) for item in data]
    elif isinstance(data, str) and len(data) > 1000:
        # Truncate very long strings
        return data[:1000] + "... [truncated]"
    return data

def mask_path(path: str) -> str:
    """
    Mask a file path to only show the filename.

    Args:
        path: File path

    Returns:
        Masked path (filename only)
    """
    if not path:
        return ""
    return os.path.basename(path)

def remove_paths_from_string(text: str) -> str:
    """
    Remove file paths from a string.

    Args:
        text: String that may contain file paths

    Returns:
        String with file paths replaced by filenames
    """
    if not text or not isinstance(text, str):
        return text

    import re

    # Common path patterns to look for
    path_patterns = [
        # Unix absolute paths
        r'(/(?:[^/\s]+/)+[^/\s]+)',
        # Windows absolute paths
        r'([A-Za-z]:\\(?:[^\\\\\/\s]+\\)+[^\\\/\s]+)',
        # Relative paths with multiple directories
        r'((?:\.\./)+(?:[^/\s]+/)+[^/\s]+)',
        r'((?:\.\\)+(?:[^\\\\\/\s]+\\)+[^\\\/\s]+)',
        # Current directory paths
        r'(\./(?:[^/\s]+/)+[^/\s]+)',
        r'(\.\\(?:[^\\\\\/\s]+\\)+[^\\\/\s]+)',
        # Paths with environment variables
        r'(\$[A-Za-z0-9_]+/(?:[^/\s]+/)+[^/\s]+)',
        r'(%[A-Za-z0-9_]+%\\(?:[^\\\\\/\s]+\\)+[^\\\/\s]+)',
        # Home directory paths
        r'(~/(?:[^/\s]+/)+[^/\s]+)',
    ]

    # Replace each path with just the filename
    for pattern in path_patterns:
        text = re.sub(pattern, lambda m: os.path.basename(m.group(1)), text)

    return text

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
LOG_FILE = os.path.join(LOG_DIR, 'application.log')
print(f"Logs will be written to: {LOG_FILE}")

# Log format for console and file
CONSOLE_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
FILE_FORMAT = '%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] [%(correlation_id)s]' + \
              '%(file_info)s%(request_info)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Custom formatter for adding file and request info
class ContextAwareFormatter(logging.Formatter):
    """
    Formatter that adds file and request context to log messages.
    Focuses on essential fields: file_id, file_type, file_size, API request, endpoint.
    """

    def format(self, record):
        # Add file info if available
        file_info = ""
        if hasattr(record, 'file_id') or hasattr(record, 'file_name') or hasattr(record, 'file_type') or hasattr(record, 'file_size_mb'):
            file_info = " [file:"
            if hasattr(record, 'file_id'):
                file_info += f" id={record.file_id}"
            if hasattr(record, 'file_name'):
                file_info += f" name={record.file_name}"
            if hasattr(record, 'file_type'):
                file_info += f" type={record.file_type}"
            if hasattr(record, 'file_size_mb'):
                file_info += f" size={record.file_size_mb}MB"
            elif hasattr(record, 'file_size'):
                file_info += f" size={record.file_size}B"
            file_info += "]"
        record.file_info = file_info

        # Add request info if available
        request_info = ""
        if hasattr(record, 'request_endpoint') or hasattr(record, 'request_method'):
            request_info = " [request:"
            if hasattr(record, 'request_method'):
                request_info += f" {record.request_method}"
            if hasattr(record, 'request_endpoint'):
                request_info += f" {record.request_endpoint}"
            request_info += "]"
        record.request_info = request_info

        return super().format(record)

class ContextFilter(logging.Filter):
    """Filter that adds context information to log records."""

    def filter(self, record):
        # Add correlation ID
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = get_correlation_id()

        # Add request context
        request_context = get_request_context()
        for key, value in request_context.items():
            setattr(record, f"request_{key}", value)

        # Add file context
        file_context = get_file_context()
        for key, value in file_context.items():
            setattr(record, f"file_{key}", value)

        return True

class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive data in log records."""

    def filter(self, record):
        # Mask sensitive data in args
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                record.args = mask_sensitive_data(record.args)
            elif isinstance(record.args, (tuple, list)):
                record.args = tuple(mask_sensitive_data(list(record.args)))

        # Mask sensitive data in message if it's a dict
        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            record.msg = mask_sensitive_data(record.msg)

        # Remove all file paths from message if it's a string
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = remove_paths_from_string(record.msg)

        # Remove file paths from any string attributes in extra data
        if hasattr(record, '__dict__'):
            for key, value in list(record.__dict__.items()):
                if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower() or 'file' in key.lower()):
                    if os.path.exists(os.path.dirname(value)):
                        # This is likely a file path, mask it
                        record.__dict__[key] = mask_path(value)

        return True

class PathFilter(logging.Filter):
    """Filter that removes file paths from log records."""

    def filter(self, record):
        # Process message
        if hasattr(record, 'msg'):
            if isinstance(record.msg, str):
                record.msg = remove_paths_from_string(record.msg)
            elif isinstance(record.msg, dict):
                # Process dictionary values
                for key, value in record.msg.items():
                    if isinstance(value, str):
                        record.msg[key] = remove_paths_from_string(value)
                    # Handle path-specific keys more aggressively
                    if isinstance(value, str) and any(path_key in key.lower() for path_key in ['path', 'dir', 'file', 'folder', 'location']):
                        record.msg[key] = mask_path(value)

        # Process args
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                for key, value in record.args.items():
                    if isinstance(value, str):
                        record.args[key] = remove_paths_from_string(value)
                    # Handle path-specific keys more aggressively
                    if isinstance(value, str) and any(path_key in key.lower() for path_key in ['path', 'dir', 'file', 'folder', 'location']):
                        record.args[key] = mask_path(value)
            elif isinstance(record.args, (tuple, list)):
                args_list = list(record.args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, str):
                        args_list[i] = remove_paths_from_string(arg)
                record.args = tuple(args_list)

        # Process all attributes in record.__dict__
        for key, value in list(record.__dict__.items()):
            # Skip internal attributes and non-string values
            if key.startswith('_') or not isinstance(value, str):
                continue

            # Handle path-specific keys more aggressively
            if any(path_key in key.lower() for path_key in ['path', 'dir', 'file', 'folder', 'location']):
                record.__dict__[key] = mask_path(value)
            else:
                # Apply general path removal to all string values
                record.__dict__[key] = remove_paths_from_string(value)

        return True

class AnsiColorStripFilter(logging.Filter):
    """Filter that strips ANSI color codes from log records."""

    def __init__(self):
        super().__init__()
        # Regex pattern to match ANSI escape codes
        import re
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def filter(self, record):
        # Strip ANSI codes from message if it's a string
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.ansi_escape.sub('', record.msg)

        # Strip ANSI codes from args if they're strings
        if hasattr(record, 'args') and record.args:
            if isinstance(record.args, dict):
                for key, value in record.args.items():
                    if isinstance(value, str):
                        record.args[key] = self.ansi_escape.sub('', value)
            elif isinstance(record.args, (tuple, list)):
                args_list = list(record.args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, str):
                        args_list[i] = self.ansi_escape.sub('', arg)
                record.args = tuple(args_list)

        return True

class ColoredFormatter(logging.Formatter):
    """
    Formatter for beautiful colorized console output.
    Only applies colors when outputting to a terminal.
    """

    # Define color schemes for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True, is_tty=None):
        super().__init__(fmt, datefmt, style, validate)
        # Determine if output is to a terminal
        self.is_tty = is_tty if is_tty is not None else sys.stdout.isatty()

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Only apply colors if output is to a terminal and colorama is available
        if self.is_tty and COLORAMA_AVAILABLE:
            levelname = record_copy.levelname

            # Apply color to the levelname based on level
            if levelname in self.COLORS:
                record_copy.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"

            # Format the record
            formatted = super().format(record_copy)

            # Apply color to the message part
            if levelname in self.COLORS:
                # Split the formatted message to color only the message part, not the timestamp
                parts = formatted.split(': ', 1)
                if len(parts) == 2:
                    formatted = f"{parts[0]}: {self.COLORS[levelname]}{parts[1]}{Style.RESET_ALL}"
                else:
                    formatted = f"{self.COLORS[levelname]}{formatted}{Style.RESET_ALL}"
        else:
            # No colors if not a terminal or colorama not available
            formatted = super().format(record_copy)

        return formatted

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    Focuses on essential fields: file_id, file_type, file_size, API request, endpoint.
    """

    def format(self, record):
        # Create a dictionary with essential log record attributes
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt or '%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', get_correlation_id())
        }

        # Add file context if available
        file_data = {}

        # Check for file_id
        if hasattr(record, 'file_id'):
            file_data['id'] = record.file_id

        # Check for file_type
        if hasattr(record, 'file_type'):
            file_data['type'] = record.file_type

        # Check for file size (in MB or bytes)
        if hasattr(record, 'file_size_mb'):
            file_data['size_mb'] = record.file_size_mb
        elif hasattr(record, 'file_size'):
            file_data['size'] = record.file_size

        # Check for file_name
        if hasattr(record, 'file_name'):
            file_data['name'] = record.file_name

        # Add file data if we have any
        if file_data:
            log_data['file'] = file_data

        # Add request context if available
        request_data = {}

        # Check for request method
        if hasattr(record, 'request_method'):
            request_data['method'] = record.request_method

        # Check for request endpoint
        if hasattr(record, 'request_endpoint'):
            request_data['endpoint'] = record.request_endpoint

        # Add request data if we have any
        if request_data:
            log_data['request'] = request_data

        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }

        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in [
                'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                'funcName', 'id', 'levelname', 'levelno', 'lineno', 'module',
                'msecs', 'message', 'msg', 'name', 'pathname', 'process',
                'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
            ] and not key.startswith('_'):
                log_data[key] = value

        # Mask sensitive data
        log_data = mask_sensitive_data(log_data)

        return json.dumps(log_data)

def configure_logging(module_name: str, async_logging=True, console_output=False):
    """
    Configure logging for a module with production-grade settings.

    Args:
        module_name: Name of the module (e.g., 'frontend.pages.analysis')
        async_logging: Whether to use asynchronous logging for better performance
        console_output: Whether to output logs to console (default: False)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(module_name)

    # Set the logging level
    level = LOG_LEVEL_MAP.get(LOG_LEVEL, logging.INFO)
    logger.setLevel(level)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters
    # Check if stdout is a terminal for proper color handling
    is_tty = sys.stdout.isatty()
    console_formatter = ColoredFormatter(CONSOLE_FORMAT, DATE_FORMAT, is_tty=is_tty)
    file_formatter = ContextAwareFormatter(FILE_FORMAT, DATE_FORMAT)

    # Add filters
    context_filter = ContextFilter()
    sensitive_filter = SensitiveDataFilter()
    ansi_strip_filter = AnsiColorStripFilter()
    path_filter = PathFilter()

    # Console handler with colored output (only if terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)

    # Add filters to console handler
    console_handler.addFilter(context_filter)
    console_handler.addFilter(sensitive_filter)
    console_handler.addFilter(path_filter)

    # File handlers with rotation
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        # Regular log file with time-based rotation - use context-aware formatter
        if async_logging:
            try:
                # Import here to avoid circular imports
                from src.shared.logging.async_handler import AsyncTimedRotatingFileHandler

                # Create async file handler
                file_handler = AsyncTimedRotatingFileHandler(
                    LOG_FILE,
                    when='midnight',
                    interval=1,
                    backupCount=30,  # Keep logs for 30 days
                    queue_size=5000  # Larger queue for high-volume logging
                )
            except ImportError:
                # Fall back to standard handler if async handler is not available
                file_handler = TimedRotatingFileHandler(
                    LOG_FILE,
                    when='midnight',
                    interval=1,
                    backupCount=30  # Keep logs for 30 days
                )
        else:
            # Standard synchronous handler
            file_handler = TimedRotatingFileHandler(
                LOG_FILE,
                when='midnight',
                interval=1,
                backupCount=30  # Keep logs for 30 days
            )

        # Set up formatters and filters for file handler
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        file_handler.addFilter(context_filter)
        file_handler.addFilter(sensitive_filter)
        file_handler.addFilter(ansi_strip_filter)  # Strip any ANSI codes from file logs
        file_handler.addFilter(path_filter)  # Remove file paths from logs

        # Add file handler to logger
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"WARNING: Could not create log file handlers: {str(e)}")

    # Only add console handler if console_output is True
    if console_output:
        logger.addHandler(console_handler)

    return logger

class ContextLogger:
    """
    Context-aware logger that adds context to log messages.
    """

    def __init__(self, logger_or_name):
        if isinstance(logger_or_name, str):
            # For backward compatibility
            self.logger = get_logger(logger_or_name, console_output=False)
        else:
            # Use the provided logger
            self.logger = logger_or_name
        self.context = {}

    def add_context(self, **kwargs):
        """Add context data that will be included in all subsequent log messages."""
        # Mask any sensitive data in context
        masked_kwargs = mask_sensitive_data(kwargs)
        self.context.update(masked_kwargs)
        return self

    def set_file_context(self, file_id=None, file_type=None, file_size=None, file_name=None):
        """Set file context for logging."""
        set_file_context(file_id, file_type, file_size, file_name)
        return self

    def set_request_context(self, method=None, endpoint=None):
        """Set request context for logging."""
        set_request_context(endpoint, method)
        return self

    def clear_context(self):
        """Clear all context data."""
        self.context = {}
        clear_context()
        return self

    def _log_with_context(self, level, msg, *args, **kwargs):
        """Log a message with the current context."""
        # Format the message with args if provided
        if args:
            msg = msg % args

        # Add context to extra data for structured logging
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra

        # Add context to the message for human-readable logs
        if self.context and isinstance(msg, str):
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

    def success(self, msg, *args, **kwargs):
        """Log a success message (uses INFO level with success indicator)."""
        self._log_with_context('info', f"✓ SUCCESS: {msg}", *args, **kwargs)
        return self

    def failure(self, msg, *args, **kwargs):
        """Log a failure message (uses ERROR level with failure indicator)."""
        self._log_with_context('error', f"✗ FAILURE: {msg}", *args, **kwargs)
        return self

def get_logger(name: str, console_output=False):
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)
        console_output: Whether to output logs to console (default: False)

    Returns:
        Logger instance
    """
    # Check if we already have a logger with this name
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Return existing logger

    # Configure a new logger
    return configure_logging(name, console_output=console_output)

def get_context_logger(name: str, console_output=False):
    """
    Get a context logger with the specified name.

    Args:
        name: Logger name (typically __name__)
        console_output: Whether to output logs to console (default: False)

    Returns:
        ContextLogger instance
    """
    # Force console_output to False for Streamlit applications to avoid duplicate logs
    # Streamlit already captures and displays console output
    logger = get_logger(name, console_output=False)
    return ContextLogger(logger)

def log_exception(logger, e, context=None):
    """
    Enhanced exception logging with full traceback and context.

    Args:
        logger: Logger instance
        e: Exception
        context: Additional context to include
    """
    if context is None:
        context = {}

    error_details = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'traceback': traceback.format_exc(),
    }

    # Add context to error details
    error_details.update(context)

    # Log the exception
    logger.exception("An error occurred", extra=error_details)

def log_file_upload(logger, file_path, file_id=None, file_type=None, file_size=None):
    """
    Log a file upload with appropriate context.

    Args:
        logger: Logger instance
        file_path: Path to the uploaded file
        file_id: Optional file ID
        file_type: Optional file type
        file_size: Optional file size in bytes
    """
    # Get file name from path
    file_name = os.path.basename(file_path)

    # Generate file ID if not provided
    if file_id is None:
        file_id = str(uuid.uuid4())[:8]

    # Calculate file size if not provided
    if file_size is None and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)

    # Guess file type if not provided
    if file_type is None:
        import mimetypes
        file_type, _ = mimetypes.guess_type(file_path)
        if file_type is None:
            file_type = "application/octet-stream"

    # Set file context
    set_file_context(
        file_id=file_id,
        file_name=file_name,
        file_type=file_type,
        file_size=file_size
    )

    # Log the upload
    logger.info(f"File uploaded: {file_name}")

    return logger

def log_file_processing(logger, file_path, operation, file_id=None):
    """
    Log a file processing operation.

    Args:
        logger: Logger instance
        file_path: Path to the file being processed
        operation: Description of the processing operation
        file_id: Optional file ID
    """
    # Get file name from path
    file_name = os.path.basename(file_path)

    # Set file context
    set_file_context(
        file_id=file_id,
        file_name=file_name
    )

    # Log the processing
    logger.info(f"Processing file: {operation}")

    return logger

def log_api_request(logger, method, endpoint, correlation_id=None):
    """
    Log an API request with appropriate context.

    Args:
        logger: Logger instance
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint path
        correlation_id: Optional correlation ID
    """
    # Set correlation ID if provided
    if correlation_id:
        set_correlation_id(correlation_id)

    # Set request context
    set_request_context(method=method, endpoint=endpoint)

    # Log the request
    logger.info(f"API request: {method} {endpoint}")

    return logger

# Initialize root logger
root_logger = logging.getLogger()
if not root_logger.handlers:
    configure_logging('root', console_output=False)