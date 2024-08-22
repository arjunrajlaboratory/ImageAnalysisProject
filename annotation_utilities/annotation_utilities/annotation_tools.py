from shapely.geometry import Point, Polygon


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
