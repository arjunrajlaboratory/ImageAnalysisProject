import functools
import re

from annotation_client.utils import sendProgress, sendWarning, sendError


def _friendly_error_message(exc):
    """Convert an exception into a user-friendly message."""
    text = str(exc)
    check_logs = "Check logs for details."

    # HTTP status code errors (from girder_client.HttpError)
    http_match = re.search(r'HTTP error (\d+)', text)
    if http_match:
        code = int(http_match.group(1))
        messages = {
            500: "Server error. " + check_logs,
            502: "Server error. " + check_logs,
            503: "Servers busy. Try again in "
                 "a few minutes.",
            504: "Servers busy. Try again in "
                 "a few minutes.",
        }
        return messages.get(
            code, f"HTTP error {code}. " + check_logs
        )

    if isinstance(exc, MemoryError):
        return "Out of memory. " + check_logs

    return f"{type(exc).__name__}. {check_logs}"


def handle_error(func):
    """Decorator that catches exceptions, sends a
    user-friendly error via sendError(), then re-raises.

    Usage:
        @handle_error
        def compute(datasetId, apiUrl, token, params):
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            msg = _friendly_error_message(exc)
            sendError(msg, info="Check logs for details.")
            raise
    return wrapper


def update_progress(processed, total, message):
    """
    Update the progress of the worker. If there are more than 100 objects,
    we only send progress every 1% of the total number of objects.
    If there are less than 100 objects, we send progress every object.
    That ensures that we never send more than 100 messages to the server,
    thereby not overwhelming the server with progress messages and flooding
    the log file.

    Args:
        processed (int): The number of objects processed.
        total (int): The total number of objects to process.
        message (str): The message to display.
    """
    if total > 100:
        if processed % int(total / 100) == 0:
            sendProgress(processed / total, message,
                         f"Processing object {processed}/{total}")
    else:
        sendProgress(processed / total, message,
                     f"Processing object {processed}/{total}")
