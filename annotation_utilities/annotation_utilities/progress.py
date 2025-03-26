from annotation_client.utils import sendProgress, sendWarning, sendError


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
