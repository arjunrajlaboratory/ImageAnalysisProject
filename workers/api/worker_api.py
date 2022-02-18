import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

import imageio

class WorkerClient:

    def __init__(self, datasetId, apiUrl, token, params):

        self.datasetId = datasetId
        self.apiUrl = apiUrl
        self.token = token
        self.params = params

        self.propertyName = params.get('customName', None)
        if not self.propertyName:
            self.propertyName = params.get('name', 'unknown_property')

        # Setup helper classes with url and credentials
        self.annotationClient = annotations.UPennContrastAnnotationClient(
            apiUrl=apiUrl, token=token)
        self.datasetClient = tiles.UPennContrastDataset(
            apiUrl=apiUrl, token=token, datasetId=datasetId)

        # Cache downloaded images by location
        self.images = {}

    def get_annotation_list(self, shape):

        annotationIds = self.params.get('annotationIds', None)

        annotationList = []
        if annotationIds:
            # Get the annotations specified by id in the parameters
            for id in annotationIds:
                annotationList.append(self.annotationClient.getAnnotationById(id))
        else:
            # Get all point annotations from the dataset
            annotationList = self.annotationClient.getAnnotationsByDatasetId(
                self.datasetId, shape=shape)

        return annotationList

    def get_image_for_annotation(self, annotation):

        # Get image location
        channel = self.params.get('channel', None)
        if channel is None:  # Default to the annotation's channel, null means Any was selected
            channel = annotation.get('channel', None)
        if channel is None:
            return None

        location = annotation['location']
        time, z, xy = location['Time'], location['Z'], location['XY']

        # Look for cached image. Initialize cache if necessary.
        image = self.images.setdefault(channel, {}).setdefault(
            time, {}).setdefault(z, {}).get(xy, None)

        if image is None:
            # Download the image at specified location
            pngBuffer = self.datasetClient.getRawImage(xy, z, time, channel)

            # Read the png buffer
            image = imageio.imread(pngBuffer)

            # Cache the image
            self.images[channel][time][z][xy] = image

        return image

    def add_annotation_property_values(self, annotation, values):

        property_values = {self.propertyName: values}

        self.annotationClient.addAnnotationPropertyValues(self.datasetId, annotation['_id'], property_values)
