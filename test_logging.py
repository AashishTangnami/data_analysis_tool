"""
Beautiful, Production-Standard Logging Test Script

This script demonstrates the colorful, informative logging system
implemented in the Data Analysis Tool.

Usage:
    python test_logging.py
"""
import os
from src.shared.logging_config import test_logging, LOG_FILE

# Try to import colorama, but don't fail if it's not available
try:
    from colorama import Fore, Back, Style
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

def create_test_log_entry():
    """Create a test log entry using direct file writing to verify file permissions."""
    try:
        with open(LOG_FILE, 'a') as f:
            f.write("\n=== DIRECT FILE WRITE TEST ===\n")
        print(f"{Fore.GREEN}✓ Successfully wrote directly to log file: {LOG_FILE}{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗ Failed to write directly to log file: {str(e)}{Style.RESET_ALL}")
        return False

if __name__ == "__main__":
    # Print colorama status
    if not COLORAMA_AVAILABLE:
        print("Note: colorama is not installed. Install it for colorized output: pip install colorama")

    # Run the logging test
    test_logging()

    # Try direct file writing as a fallback test
    print(f"\n{Fore.YELLOW}Testing Direct File Writing:{Style.RESET_ALL}")
    create_test_log_entry()

    # Print file location
    print(f"\n{Fore.GREEN}Logs are being saved to: {LOG_FILE}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}You can view them with: cat {LOG_FILE}{Style.RESET_ALL}")
