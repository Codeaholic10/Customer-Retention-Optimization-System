import sys
import types


def error_message_detail(error: Exception, error_detail: types.ModuleType) -> str:
    """
    Extract detailed error information including file name and line number.

    Parameters
    ----------
    error : Exception
        The exception object.
    error_detail : types.ModuleType
        The ``sys`` module, used to access the current traceback via
        ``sys.exc_info()``.

    Returns
    -------
    str
        Formatted error message with script name and line number.
        Falls back gracefully when called outside an active except-block
        (i.e. when ``sys.exc_info()`` returns ``(None, None, None)``).
    """
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is None:
        # Called outside an active exception context – best-effort message
        return (
            "Error (no traceback available): [{0}]".format(str(error))
        )

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
    Custom exception that enriches the error message with the source file
    path and line number where the exception originally occurred.

    Usage
    -----
    .. code-block:: python

        try:
            ...
        except Exception as exc:
            raise CustomException(exc, sys) from exc
    """

    def __init__(self, error_message: Exception, error_detail: types.ModuleType) -> None:
        """
        Parameters
        ----------
        error_message : Exception
            The original exception (or a plain string message).
        error_detail : types.ModuleType
            Pass the ``sys`` module so the traceback can be captured.
        """
        super().__init__(error_message)
        self.error_message: str = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        return self.error_message
