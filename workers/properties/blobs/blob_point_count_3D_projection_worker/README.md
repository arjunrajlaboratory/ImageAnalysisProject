# sample_intensity_worker
Sample computed property worker for the Kitware/UPennContrast project

This example file uses ITK to compute average intensity values in a radius around point annotations.

Makes use of the utility client for communication with the annotation plugin. The client is available in UPennContrast/devops/girder/annotation_client.

It is mandatory to specify an entrypoint in the Dockerfile. Anything else is optional though python is recommended for now as it allow usage of the annotation client.

The following arguments are passed to the entrypoint:

* `--apiUrl`: the url to the girder API
* `--token`: authentication token for girder
* `--datasetId`: the id of the dataset to process
* `--parameters`: json-formatted string containing configuration information related to the property
