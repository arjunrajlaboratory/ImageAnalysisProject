import argparse
import json
import sys
import time

import annotation_client.workers as workers
from annotation_client.utils import sendProgress, sendWarning, sendError
from annotation_utilities.progress import handle_error


def interface(image, apiUrl, token):
    client = workers.UPennContrastWorkerPreviewClient(
        apiUrl=apiUrl, token=token)

    interface = {
        'Error Generator': {
            'type': 'notes',
            'value': ('Generates various error conditions '
                      'for testing frontend error reporting. '
                      'Check the boxes below to trigger '
                      'specific error types.'),
            'displayOrder': 0,
        },
        'Send warning message': {
            'type': 'checkbox',
            'default': False,
            'tooltip': ('Sends a warning via sendWarning() '
                        '— worker continues running.'),
            'displayOrder': 1,
        },
        'Send error message': {
            'type': 'checkbox',
            'default': False,
            'tooltip': ('Sends an error via sendError() '
                        '— worker continues running.'),
            'displayOrder': 2,
        },
        'Crash immediately': {
            'type': 'checkbox',
            'default': False,
            'tooltip': ('Raises an unhandled exception '
                        'immediately (no progress shown).'),
            'displayOrder': 3,
        },
        'Crash after progress': {
            'type': 'checkbox',
            'default': False,
            'tooltip': ('Shows progress updates then crashes '
                        'mid-way with a traceback.'),
            'displayOrder': 4,
        },
        'Simulate HTTP 500': {
            'type': 'checkbox',
            'default': False,
            'tooltip': ('Simulates an HTTP 500 Internal Server Error '
                        'traceback like girder_client.HttpError.'),
            'displayOrder': 5,
        },
        'Simulate HTTP 503': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Simulates an HTTP 503 Service Unavailable error.',
            'displayOrder': 6,
        },
        'Simulate HTTP 504': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Simulates an HTTP 504 Gateway Timeout error.',
            'displayOrder': 7,
        },
        'Simulate OOM kill': {
            'type': 'checkbox',
            'default': False,
            'tooltip': 'Simulates an out-of-memory crash (MemoryError).',
            'displayOrder': 8,
        },
        'Delay before error (seconds)': {
            'type': 'number',
            'min': 0,
            'max': 30,
            'default': 1,
            'tooltip': ('How long to wait before triggering '
                        'each error condition.'),
            'unit': 'seconds',
            'displayOrder': 9,
        },
    }
    client.setWorkerImageInterface(image, interface)


@handle_error
def compute(datasetId, apiUrl, token, params):
    workerInterface = params.get('workerInterface', {})

    delay = float(
        workerInterface.get('Delay before error (seconds)', 1)
    )

    sendProgress(0.0, 'Error Generator', 'Starting error generation...')
    time.sleep(delay)

    step = 0
    total_steps = sum([
        workerInterface.get('Send warning message', False),
        workerInterface.get('Send error message', False),
        workerInterface.get('Crash immediately', False),
        workerInterface.get('Crash after progress', False),
        workerInterface.get('Simulate HTTP 500', False),
        workerInterface.get('Simulate HTTP 503', False),
        workerInterface.get('Simulate HTTP 504', False),
        workerInterface.get('Simulate OOM kill', False),
    ])

    if total_steps == 0:
        sendProgress(
            1.0, 'Error Generator',
            'No error types selected. Nothing to do.'
        )
        return

    # --- Send warning message ---
    if workerInterface.get('Send warning message', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Triggering warning message ({step}/{total_steps})')
        time.sleep(delay)
        sendWarning(
            'Sample warning from Error Generator',
            info='This is an intentional warning message '
                 'for testing frontend error reporting.'
        )

    # --- Send error message ---
    if workerInterface.get('Send error message', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Triggering error message ({step}/{total_steps})')
        time.sleep(delay)
        sendError(
            'Sample error from Error Generator',
            info='This is an intentional error message for '
                 'testing frontend error reporting. '
                 'The worker will continue running after this.'
        )

    # --- Crash immediately ---
    if workerInterface.get('Crash immediately', False):
        raise RuntimeError(
            "Intentional crash from Error Generator worker. "
            "This simulates an unhandled exception with no prior progress."
        )

    # --- Crash after progress ---
    if workerInterface.get('Crash after progress', False):
        # Show some realistic progress before crashing
        for i in range(5):
            frac = (i + 1) / 15
            sendProgress(
                frac, 'Processing frames',
                f'Processing frame {i+1}/15'
            )
            time.sleep(delay * 0.3)

        raise Exception(
            "Simulated crash after progress. "
            "This mimics a worker that fails mid-computation, "
            "e.g. due to a corrupted image tile or unexpected data format."
        )

    # --- Simulate HTTP 500 ---
    if workerInterface.get('Simulate HTTP 500', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Simulating HTTP 500 ({step}/{total_steps})')
        # Show some progress first, like a real worker would
        for i in range(3):
            frac = 0.05 * (i + 1)
            sendProgress(
                frac, 'Processing',
                f'Processing frame {i+1}/15'
            )
            time.sleep(delay * 0.3)

        # Simulate the traceback from girder_client
        raise Exception(
            "HTTP error 500: GET http://172.16.0.4/"
            "girder/api/v1//item/69c412afdce95c31e3675c50"
            "/tiles/region?frame=3&encoding=pickle%3A5\n"
            "Response text: {\"message\": "
            "\"An unexpected error occurred on the "
            "server.\", \"type\": \"internal\", "
            "\"uid\": \"626d0474-15f0-47b8-87a1-"
            "184ac9ec4ba5\"}"
        )

    # --- Simulate HTTP 503 ---
    if workerInterface.get('Simulate HTTP 503', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Simulating HTTP 503 ({step}/{total_steps})')
        # Show extended progress like a long-running property worker
        for i in range(8):
            frac = 0.05 * (i + 1)
            sendProgress(frac, 'Computing blob intensity',
                         f'Processing object {(i+1)*500}/5000')
            time.sleep(delay * 0.2)

        raise Exception(
            "HTTP error 503: GET http://172.16.0.4/"
            "girder/api/v1//item/"
            "69aa4ca15aeb4ca4a398ce30/tiles/region"
            "?frame=418&encoding=pickle%3A5\n"
            "Response text: <html><body>"
            "<h1>503 Service Unavailable</h1>\n"
            "No server is available to handle "
            "this request.\n</body></html>"
        )

    # --- Simulate HTTP 504 ---
    if workerInterface.get('Simulate HTTP 504', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Simulating HTTP 504 ({step}/{total_steps})')
        # Show lots of progress like a worker near completion
        for i in range(10):
            frac = 0.3 + 0.06 * (i + 1)
            sendProgress(frac, 'Uploading property values',
                         f'Processed {(i+1)*10000}/100000 connections')
            time.sleep(delay * 0.2)

        raise Exception(
            "HTTP error 504: DELETE http://172.16.0.4/"
            "girder/api/v1//"
            "annotation_property_values"
            "?propertyId=69bffdf6b8e2b96cdb291c24"
            "&datasetId=69a85d49dd9566b7e3b928e5\n"
            "Response text: <html><body>"
            "<h1>504 Gateway Time-out</h1>\n"
            "The server didn't respond in time.\n"
            "</body></html>"
        )

    # --- Simulate OOM kill ---
    if workerInterface.get('Simulate OOM kill', False):
        step += 1
        sendProgress(step / total_steps * 0.9, 'Error Generator',
                     f'Simulating OOM kill ({step}/{total_steps})')
        time.sleep(delay)
        raise MemoryError(
            "Simulated out-of-memory error. "
            "This mimics what happens when a worker "
            "exhausts available memory, e.g. loading "
            "a very large image into RAM."
        )

    sendProgress(
        1.0, 'Error Generator',
        'All selected error conditions triggered.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate various error conditions for frontend testing')

    parser.add_argument('--datasetId', type=str,
                        required=False, action='store')
    parser.add_argument('--apiUrl', type=str, required=True, action='store')
    parser.add_argument('--token', type=str, required=True, action='store')
    parser.add_argument('--request', type=str, required=True, action='store')
    parser.add_argument('--parameters', type=str,
                        required=True, action='store')

    args = parser.parse_args(sys.argv[1:])

    params = json.loads(args.parameters)
    datasetId = args.datasetId
    apiUrl = args.apiUrl
    token = args.token

    match args.request:
        case 'compute':
            compute(datasetId, apiUrl, token, params)
        case 'interface':
            interface(params['image'], apiUrl, token)
