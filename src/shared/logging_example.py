"""
Example usage of the logging system.

This demonstrates how to log:
- File information (ID, type, size)
- API request information (method, endpoint)
- Success and error messages
"""

from src.shared.logging_config import get_context_logger, set_correlation_id

def example_file_upload():
    """Example of logging a file upload."""
    # Create a logger
    logger = get_context_logger("file_upload_example")

    # Set a correlation ID for this operation
    correlation_id = set_correlation_id("upload-123456")

    # Log the start of the operation
    logger.info(f"Starting file upload operation with correlation ID: {correlation_id}")

    # Set file context
    logger.set_file_context(
        file_id="file-123",
        file_name="example.csv",
        file_type="text/csv",
        file_size=1024 * 1024 * 5  # 5MB
    )

    # Log file processing
    logger.info("Processing file")

    try:
        # Simulate successful processing
        logger.success("File processed successfully")
    except Exception as e:
        # Log any errors
        logger.exception("Error processing file")
        logger.failure(f"File processing failed: {str(e)}")

def example_api_request():
    """Example of logging an API request."""
    # Create a logger
    logger = get_context_logger("api_example")

    # Set a correlation ID for this request
    correlation_id = set_correlation_id("request-789012")

    # Set request context
    logger.set_request_context(
        method="POST",
        endpoint="/api/data/analyze"
    )

    # Log the request
    logger.info(f"Received API request with correlation ID: {correlation_id}")

    # Set file context if the request involves a file
    logger.set_file_context(
        file_id="file-456",
        file_name="data.json",
        file_type="application/json",
        file_size=2048  # 2KB
    )

    # Log processing steps
    logger.info("Validating request parameters")
    logger.info("Processing request")

    # Log success
    logger.success("API request processed successfully")

    # Clear context when done
    logger.clear_context()

if __name__ == "__main__":
    # Run the examples
    example_file_upload()
    example_api_request()
