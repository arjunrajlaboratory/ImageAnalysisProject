import numpy as np

from itertools import product
from math import prod
from operator import itemgetter
from shapely.geometry import Polygon
from typing import Sequence

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

from annotation_client.utils import sendProgress
from annotation_utilities import batch_argument_parser


class WorkerClient:

    def __init__(self, datasetId, apiUrl, token, params):

        self.datasetId = datasetId
        self.apiUrl = apiUrl
        self.token = token
        self.params = params

        # roughly validate params
        keys = ["assignment", "channel", "connectTo",
                "tags", "tile", "workerInterface"]
        if not all(key in params for key in keys):
            print("Invalid worker parameters", params)
            return
        assignment, channel, connectTo, tags, tile, workerInterface = itemgetter(
            *keys)(params)

        self.assignment = assignment
        self.channel = channel
        self.connectTo = connectTo
        self.tags = tags
        self.tile = tile
        self.workerInterface = workerInterface

        batch_xy = workerInterface.get('Batch XY', None)
        batch_z = workerInterface.get('Batch Z', None)
        batch_time = workerInterface.get('Batch Time', None)

        batch_xy = batch_argument_parser.process_range_list(
            batch_xy, convert_one_to_zero_index=True)
        batch_z = batch_argument_parser.process_range_list(
            batch_z, convert_one_to_zero_index=True)
        batch_time = batch_argument_parser.process_range_list(
            batch_time, convert_one_to_zero_index=True)

        if batch_xy is None:
            batch_xy = [tile['XY']]
        if batch_z is None:
            batch_z = [tile['Z']]
        if batch_time is None:
            batch_time = [tile['Time']]

        self.batch_xy = batch_xy
        self.batch_z = batch_z
        self.batch_time = batch_time

        annotationClient = annotations.UPennContrastAnnotationClient(
            apiUrl=apiUrl, token=token)
        datasetClient = tiles.UPennContrastDataset(
            apiUrl=apiUrl, token=token, datasetId=datasetId)

        self.annotationClient = annotationClient
        self.datasetClient = datasetClient

    def get_image(self, xy=None, z=None, time=None, channel=None):

        if xy is None:
            xy = self.tile['XY']
        if z is None:
            z = self.tile['Z']
        if time is None:
            time = self.tile['Time']
        if channel is None:
            channel = self.channel

        frame = self.datasetClient.coordinatesToFrameIndex(
            xy, z, time, channel)
        image = self.datasetClient.getRegion(
            self.datasetId, frame=frame).squeeze()

        return image

    def get_image_stack(self, location, stack_xys=None, stack_zs=None, stack_times=None, stack_channels=None):

        xy, z, time, channel = location

        if stack_xys == 'all':
            xys = range(
                self.datasetClient.tiles['IndexRange'].get('IndexXY', 0))
        elif isinstance(stack_xys, Sequence) and len(stack_xys):
            xys = stack_xys
        else:
            if xy is None:
                xys = [self.tile['XY']]
            else:
                xys = [xy]

        if stack_zs == 'all':
            zs = range(self.datasetClient.tiles['IndexRange'].get('IndexZ', 0))
        elif isinstance(stack_zs, Sequence) and len(stack_zs):
            zs = stack_zs
        else:
            if z is None:
                zs = [self.tile['Z']]
            else:
                zs = [z]

        if stack_times == 'all':
            times = range(
                self.datasetClient.tiles['IndexRange'].get('IndexT', 0))
        elif isinstance(stack_times, Sequence) and len(stack_times):
            times = stack_times
        else:
            if time is None:
                times = [self.tile['Time']]
            else:
                times = [time]

        if stack_channels == 'all':
            channels = range(
                self.datasetClient.tiles['IndexRange'].get('IndexC', 0))
        elif isinstance(stack_channels, Sequence) and len(stack_channels):
            channels = stack_channels
        else:
            if channel is None:
                channels = [self.channel]
            else:
                channels = [channel]

        shape = (l for l, s in zip((len(xys), len(zs), len(times), len(channels)), (stack_xys,
                 stack_zs, stack_times, stack_channels)) if isinstance(s, Sequence) and len(s))

        frames = []

        for xy, z, time, channel in product(xys, zs, times, channels):
            image = self.get_image(xy, z, time, channel)
            frames.append(image)

        image_stack = np.stack(frames)
        image_stack = image_stack.reshape(*shape, *image_stack.shape[-2:])

        return image_stack

    def create_point_annotations(self, location, coords):

        xy, z, time, channel = location

        ndim = coords.shape[-1]

        annotation_template = {
            "tags": self.tags,
            "shape": "point",
            "channel": channel,
            "datasetId": self.datasetId
        }

        print(f"Uploading {len(coords)} annotations")
        annotation_list = []

        if ndim == 2:

            annotation_template = annotation_template | {
                "location": {
                    "XY": xy,
                    "Z": z,
                    "Time": time
                },
            }

            for [y, x] in coords:
                annotation = annotation_template | {
                    "coordinates": [{"x": float(x), "y": float(y), "z": float(z)}]
                }
                annotation_list.append(annotation)

        elif ndim == 3:

            for [z, y, x] in coords:
                annotation = annotation_template | {
                    "location": {
                        "XY": xy,
                        "Z": int(z),
                        "Time": time
                    },
                    "coordinates": [{"x": float(x), "y": float(y), "z": float(z)}]
                }
                annotation_list.append(annotation)

        annotationsIds = [
            a['_id'] for a in self.annotationClient.createMultipleAnnotations(annotation_list)]
        if len(self.connectTo['tags']) > 0:
            self.annotationClient.connectToNearest(
                self.connectTo, annotationsIds)

    def create_polygon_annotations(self, location, polygons):

        xy, z, time, channel = location

        annotation_template = {
            "tags": self.tags,
            "shape": "polygon",
            "channel": channel,
            "location": {
                "XY": xy,
                "Z": z,
                "Time": time
            },
            "datasetId": self.datasetId
        }

        print(f"Uploading {len(polygons)} annotations")
        annotation_list = []

        for polygon in polygons:
            polygon = Polygon(polygon)
            polygon_coords = list(polygon.exterior.coords)
            annotation = annotation_template | {
                "coordinates": [{"x": float(x), "y": float(y), "z": float(z)} for x, y in polygon_coords]
            }
            annotation_list.append(annotation)

        annotationsIds = [
            a['_id'] for a in self.annotationClient.createMultipleAnnotations(annotation_list)]
        if len(self.connectTo['tags']) > 0:
            self.annotationClient.connectToNearest(
                self.connectTo, annotationsIds)

    def process(self, f_process, f_annotation, stack_xys=None, stack_zs=None, stack_times=None, stack_channels=None,
                progress_text='Running Worker'):

        if f_annotation == 'point':
            f_annotation = self.create_point_annotations
        elif f_annotation == 'polygon':
            f_annotation = self.create_polygon_annotations

        batch = []
        if stack_xys is None:
            batch.append(list(self.batch_xy))
        else:
            batch.append([self.tile['XY']])
        if stack_zs is None:
            batch.append(list(self.batch_z))
        else:
            batch.append([self.tile['Z']])
        if stack_times is None:
            batch.append(list(self.batch_time))
        else:
            batch.append([self.tile['Time']])
        batch.append([self.channel])

        steps = prod((len(b) for b in batch))
        step = 0

        for xy, z, time, channel in product(*batch):

            image = self.get_image_stack(
                (xy, z, time, channel), stack_xys, stack_zs, stack_times, stack_channels)

            output = f_process(image)

            f_annotation((xy, z, time, channel), output)

            step += 1

            sendProgress(step / steps, progress_text, f"{step}/{steps}")
