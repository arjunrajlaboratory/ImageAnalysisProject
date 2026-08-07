import numpy as np

from itertools import product
from math import prod
from operator import itemgetter
from shapely.geometry import Polygon
from typing import Sequence

import annotation_client.annotations as annotations
import annotation_client.tiles as tiles

from annotation_client.utils import sendError, sendProgress
from annotation_utilities import batch_argument_parser
# Re-exported for workers that import it from worker_client (e.g. cellposesam).
from annotation_utilities.annotation_tools import geometry_to_polygon_coords


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

        annotationClient = annotations.UPennContrastAnnotationClient(
            apiUrl=apiUrl, token=token)
        datasetClient = tiles.UPennContrastDataset(
            apiUrl=apiUrl, token=token, datasetId=datasetId)

        self.annotationClient = annotationClient
        self.datasetClient = datasetClient

        # Batch parsing happens after datasetClient exists because 'all'
        # expands from the dataset's IndexRange. get_batch_ranges materializes
        # the one-shot generators into lists so validation and processing can
        # inspect the same coordinates, and names the offending field when the
        # input is malformed.
        index_range = datasetClient.tiles.get('IndexRange', {})
        try:
            self.batch_xy, self.batch_z, self.batch_time = (
                batch_argument_parser.get_batch_ranges(
                    tile, workerInterface, index_range)
            )
        except ValueError as exc:
            sendError("Could not read the batch range.", info=str(exc))
            raise

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

    def validate_coordinates(self, stack_xys=None, stack_zs=None,
                             stack_times=None, stack_channels=None):
        """Reject image coordinates that do not exist in the dataset.

        Batch fields are 1-indexed in the UI but are stored on this client as
        zero-indexed coordinates. Stack selections and channels are already
        zero-indexed. A stacked dimension supersedes its corresponding Batch
        field, matching :meth:`process` and :meth:`get_image_stack`.
        """
        index_range = self.datasetClient.tiles.get('IndexRange', {})

        dimensions = (
            ('Batch XY', 'XY', 'IndexXY', self.batch_xy,
             stack_xys, self.tile['XY']),
            ('Batch Z', 'Z', 'IndexZ', self.batch_z,
             stack_zs, self.tile['Z']),
            ('Batch Time', 'Time', 'IndexT', self.batch_time,
             stack_times, self.tile['Time']),
        )
        errors = []

        for field, label, index_key, batch_values, stack_values, tile_value in dimensions:
            if stack_values == 'all':
                continue
            if (isinstance(stack_values, Sequence)
                    and not isinstance(stack_values, (str, bytes))
                    and len(stack_values)):
                values = stack_values
            elif stack_values is None:
                values = batch_values
            else:
                # Empty stack selections fall back to the current tile in
                # get_image_stack().
                values = [tile_value]

            size = index_range.get(index_key, 1)
            invalid = sorted(set(value for value in values
                                 if value < 0 or value >= size))
            if invalid:
                positions = [value + 1 for value in invalid]
                position_word = 'position' if len(positions) == 1 else 'positions'
                dataset_word = 'position' if size == 1 else 'positions'
                errors.append(
                    f"{field} contains invalid {position_word} "
                    f"{', '.join(str(value) for value in positions)}. Batch "
                    f"positions start at 1; this dataset has {size} {label} "
                    f"{dataset_word}, so its valid range is 1-{size}.")

        if errors:
            detail = ' '.join(errors)
            sendError("Batch range is out of bounds.", info=detail)
            raise ValueError(detail)

        if stack_channels == 'all':
            return
        if (isinstance(stack_channels, Sequence)
                and not isinstance(stack_channels, (str, bytes))
                and len(stack_channels)):
            channels = stack_channels
        else:
            channels = [self.channel]

        num_channels = index_range.get('IndexC', 1)
        invalid_channels = sorted(set(
            channel for channel in channels
            if channel < 0 or channel >= num_channels))
        if invalid_channels:
            index_word = 'index' if len(invalid_channels) == 1 else 'indices'
            channel_word = 'channel' if num_channels == 1 else 'channels'
            detail = (
                f"The selected channel {index_word} "
                f"{', '.join(str(channel) for channel in invalid_channels)} "
                f"{'does' if len(invalid_channels) == 1 else 'do'} not exist in "
                f"this dataset, which has {num_channels} {channel_word} "
                f"(valid indices: 0-{num_channels - 1}).")
            sendError("Selected channels are outside this dataset.", info=detail)
            raise ValueError(detail)

    def get_image_stack(self, location, stack_xys=None, stack_zs=None, stack_times=None, stack_channels=None):

        xy, z, time, channel = location

        index_range = self.datasetClient.tiles.get('IndexRange', {})

        if stack_xys == 'all':
            xys = range(index_range.get('IndexXY', 1))
        elif isinstance(stack_xys, Sequence) and len(stack_xys):
            xys = stack_xys
        else:
            if xy is None:
                xys = [self.tile['XY']]
            else:
                xys = [xy]

        if stack_zs == 'all':
            zs = range(index_range.get('IndexZ', 1))
        elif isinstance(stack_zs, Sequence) and len(stack_zs):
            zs = stack_zs
        else:
            if z is None:
                zs = [self.tile['Z']]
            else:
                zs = [z]

        if stack_times == 'all':
            times = range(index_range.get('IndexT', 1))
        elif isinstance(stack_times, Sequence) and len(stack_times):
            times = stack_times
        else:
            if time is None:
                times = [self.tile['Time']]
            else:
                times = [time]

        if stack_channels == 'all':
            channels = range(index_range.get('IndexC', 1))
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

        annotation_list = []
        skipped = 0

        for polygon in polygons:
            try:
                geom = Polygon(polygon)
            except (ValueError, TypeError):
                # Fewer than the 3 distinct vertices a polygon requires.
                skipped += 1
                continue
            coord_lists = geometry_to_polygon_coords(geom)
            if not coord_lists:
                # Empty / zero-area geometry (e.g. shrunk away by negative padding).
                skipped += 1
                continue
            for polygon_coords in coord_lists:
                annotation = annotation_template | {
                    "coordinates": [{"x": float(x), "y": float(y), "z": float(z)} for x, y in polygon_coords]
                }
                annotation_list.append(annotation)

        if skipped:
            print(f"Skipped {skipped} degenerate polygon(s) (empty or zero-area)")
        print(f"Uploading {len(annotation_list)} annotations")
        if not annotation_list:
            # Posting an empty coordinates payload triggers a server 400, and an
            # empty batch has nothing to upload, so there is nothing to do.
            return

        annotationsIds = [
            a['_id'] for a in self.annotationClient.createMultipleAnnotations(annotation_list)]
        if len(self.connectTo['tags']) > 0:
            self.annotationClient.connectToNearest(
                self.connectTo, annotationsIds)

    def process(self, f_process, f_annotation, stack_xys=None, stack_zs=None, stack_times=None, stack_channels=None,
                progress_text='Running Worker'):

        self.validate_coordinates(stack_xys, stack_zs,
                                  stack_times, stack_channels)

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
