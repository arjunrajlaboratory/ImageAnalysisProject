from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.colors as mcolors


def create_points_from_annotations(elements):
    """
    Create a list of Point objects from the x and y coordinates in each dictionary element.

    Args:
    - elements: a list of dictionary elements containing x and y coordinates

    Returns:
    - points: a list of Point objects created from the x and y coordinates in each dictionary element
    """
    points = []
    for element in elements:
        coords = element['coordinates'][0]  # Assume there is only one coordinate in the list
        x, y = coords['x'], coords['y']
        point = Point(x, y)
        points.append(point)
    return points

def filter_elements_T_XY(elements, time_value, xy_value):
    return [element for element in elements if element['location']['Time'] == time_value and element['location']['XY'] == xy_value]

def filter_elements_T_XY_Z(elements, time_value, xy_value, z_value):
    return [element for element in elements if element['location']['Time'] == time_value and element['location']['XY'] == xy_value and element['location']['Z'] == z_value]

def get_annotations_with_tags(elements, tags, exclusive=False):
    result = []
    tags_set = set(tags)
    for element in elements:
        element_tags_set = set(element.get('tags', []))
        if exclusive:
            # only add the element if its tags exactly match the provided tags
            if element_tags_set == tags_set:
                result.append(element)
        else:
            # add the element if it contains any of the provided tags
            if tags_set & element_tags_set:
                result.append(element)
    return result

def get_annotations_with_tag(elements, tag, exclusive=False):
    result = []
    for element in elements:
        if exclusive:
            if element.get('tags') == [tag]:
                result.append(element)
        else:
            if tag in element.get('tags'):
                result.append(element)
    return result


def find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=True):
    """
    This function filters the target_list based on the 'location' of the source point.
    The function parameters 'Time', 'XY', and 'Z' can be set to True or False to specify whether these 'location' attributes need to be matched.
    By default, all of these parameters are set to True, meaning all 'location' attributes need to match.

    Parameters:
    source (dict): The source point annotation object
    target_list (list): The list of target point annotation objects
    Time (bool): Specifies whether the 'Time' attribute of 'location' needs to be matched. Default is True.
    XY (bool): Specifies whether the 'XY' attribute of 'location' needs to be matched. Default is True.
    Z (bool): Specifies whether the 'Z' attribute of 'location' needs to be matched. Default is True.

    Returns:
    list: The filtered list of target point annotation objects that match the specified 'location' attributes

    Example of usage:
    1) Matching all 'location' attributes:
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list)

    2) Matching specified 'location' attributes (in this case, 'Time' and 'XY'):
    source = {...}  # source point annotation object
    target_list = [...]  # target point annotation list
    matching_annotations = find_matching_annotations_by_location(source, target_list, Time=True, XY=True, Z=False)
    """
    params = {'Time': Time, 'XY': XY, 'Z': Z}
    return [target for target in target_list if all(source['location'].get(attr) == target['location'].get(attr) for attr, value in params.items() if value)]

def annotations_to_polygons(annotations):
    """
    Convert annotations to shapely Polygon objects.
    
    Args:
    annotations (list or dict): A single annotation dictionary or a list of annotation dictionaries.
    
    Returns:
    list: A list of shapely Polygon objects.
    """
    if isinstance(annotations, dict):
        annotations = [annotations]
    
    polygons = []
    for annotation in annotations:
        coords = [(point['x'], point['y']) for point in annotation['coordinates']]
        polygons.append(Polygon(coords))
    
    return polygons

def polygons_to_annotations(polygons, datasetId, XY=0, Time=0, Z=0, tags=None, channel=0):
    """
    Convert shapely Polygon objects to a list of annotations.
    
    Args:
    polygons (list): A list of shapely Polygon objects.
    XY (int): The XY position for all annotations. Default is 0.
    Time (int): The Time position for all annotations. Default is 0.
    Z (int): The Z position for all annotations. Default is 0.
    tags (list): A list of tags to apply to all annotations. Default is None.
    channel (int): The channel for all annotations. Default is 0.
    datasetId (str): The datasetId for all annotations.

    Returns:
    list: A list of annotation dictionaries.
    """
    if not isinstance(polygons, list):
        polygons = [polygons]
    
    annotations = []
    for polygon in polygons:
        coordinates = [{'x': float(x), 'y': float(y)} for x, y in list(polygon.exterior.coords)[:-1]]  # Exclude the last point as it's the same as the first
        
        annotation = {
            'coordinates': coordinates,
            'location': {'XY': XY, 'Time': Time, 'Z': Z},
            'shape': 'polygon',
            'channel': channel,
            'datasetId': datasetId
        }
        
        if tags:
            annotation['tags'] = tags
        
        annotations.append(annotation)
    
    return annotations

def get_images_for_all_channels(tileClient, datasetId, XY, Z, Time):
    """
    Get images for all channels for a given XY, Z, Time
    Returns a list of images, one for each channel
    """
    images = []
    for channel in range(0, tileClient.tiles['IndexRange']['IndexC']):
        frame = tileClient.coordinatesToFrameIndex(XY, Z, Time, channel)
        image = tileClient.getRegion(datasetId, frame=frame)
        images.append(image)
    return images

def get_layers(GirderClient, datasetId):
    """
    This function takes a datasetId and a client, and returns the layers 
    with information about contrast settings that are currently being applied.
    
    Note: A dataset can belong to multiple configurations, so there is some ambiguity here.
    The function takes the first configuration it finds. To do this properly would require 
    extensive reworking, because the front end and worker interface would all have to change 
    to pass the configurationId along with the datasetId. The user will also have to save 
    their contrast settings in the user interface in order for them to be detected in this way.
    """
    configurations = GirderClient.get("dataset_view", parameters={'datasetId': datasetId})
    configurationId = configurations[0]['configurationId']
    configuration = GirderClient.get("item/" + configurationId)
    layers = configuration['meta']['layers']
    return layers

def process_and_merge_channels(images, layers, mode='lighten'):
    layers = sorted(layers, key=lambda x: x['channel'])
    processed_channels = []
    
    for img, layer in zip(images, layers):
        if layer['visible'] == False:
            continue
        img = np.squeeze(img)
        
        contrast_mode = layer['contrast']['mode']
        black_point = layer['contrast']['blackPoint']
        white_point = layer['contrast']['whitePoint']
        
        if contrast_mode == 'percentile':
            black_value = np.percentile(img, black_point)
            white_value = np.percentile(img, white_point)
        elif contrast_mode == 'absolute':
            black_value = black_point
            white_value = white_point
        else:
            raise ValueError(f"Unsupported contrast mode: {contrast_mode}")
        
        print(f"Channel {layer['channel']} - black_value: {black_value}, white_value: {white_value}")
        
        img_normalized = np.clip((img - black_value) / (white_value - black_value), 0, 1)
        
        color = np.array(mcolors.to_rgb(layer['color']))
        img_colored = img_normalized[:,:,np.newaxis] * color
        
        processed_channels.append(img_colored)
    
    if mode == 'lighten':
        merged_image = np.max(processed_channels, axis=0)
    elif mode == 'add':
        merged_image = np.sum(processed_channels, axis=0)
        merged_image = np.clip(merged_image, 0, 1)
    elif mode == 'screen':
        merged_image = 1 - np.prod(1 - np.array(processed_channels), axis=0)
    else:
        raise ValueError("Unsupported mode. Choose 'lighten', 'add', or 'screen'.")
    
    return merged_image
