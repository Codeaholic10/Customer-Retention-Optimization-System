import sys


def error_message_detail(error, error_detail: sys):
    """
    Extract detailed error information including file name and line number.

    Args:
        error: The exception object.
        error_detail: sys module to access traceback info.

    Returns:
        Formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    error_message = (
        "Error occurred in Python script: [{0}] "
        "at line number: [{1}] "
        "with error message: [{2}]"
    ).format(file_name, line_number, str(error))

    return error_message


class CustomException(Exception):
    """
    Custom exception class that provides detailed error information,
    including the script name and line number where the error occurred.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Initialize CustomException.

        Args:
            error_message: The error message or original exception.
            error_detail: sys module to extract traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
