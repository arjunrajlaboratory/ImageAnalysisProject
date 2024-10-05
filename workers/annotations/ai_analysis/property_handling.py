import pandas as pd

def flatten_properties(properties):
    flat_properties = {}
    for key, value in properties.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_properties[f"{key}_{subkey}"] = subvalue
        else:
            flat_properties[key] = value
    return flat_properties

def get_annotation_properties(annotation_id):
    return combined_data.get(annotation_id, {}).get('values', {})

def get_property_for_all_annotations(property_name):
    return df[property_name].dropna()

# Function to create mappings from property IDs to property names and vice versa
def create_property_mappings(property_descriptions):
    """
    Creates two dictionaries for mapping property IDs to names and names to IDs.

    Args:
        property_descriptions (list): A list of property description dictionaries.

    Returns:
        tuple: Two dictionaries (property_id_to_name, property_name_to_id)
    """
    property_id_to_name = {}
    property_name_to_id = {}
    for prop in property_descriptions:
        prop_id = prop['_id']
        prop_name = prop['name']
        property_id_to_name[prop_id] = prop_name
        property_name_to_id[prop_name] = prop_id
    return property_id_to_name, property_name_to_id

# Function to flatten nested property values
def flatten_properties(properties, parent_key=''):
    """
    Flattens nested dictionaries into a single-level dictionary with compound keys.

    Args:
        properties (dict): The nested dictionary to flatten.
        parent_key (str): The base key string for compound keys.

    Returns:
        dict: A flattened dictionary.
    """
    items = {}
    for key, value in properties.items():
        new_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_properties(value, new_key))
        else:
            items[new_key] = value
    return items

# Function to combine property descriptions and values into a unified structure
def combine_property_data(property_values, property_id_to_name, annotation_list):
    """
    Combines property values with their corresponding property names and includes annotation tags.

    Args:
        property_values (list): A list of property value dictionaries.
        property_id_to_name (dict): A mapping from property IDs to names.
        annotation_list (list): A list of annotation dictionaries.

    Returns:
        dict: A dictionary with annotation IDs as keys and property values as values.
    """
    # Create a mapping from annotation IDs to their tags
    annotation_id_to_tags = {
        annotation['_id']: annotation.get('tags', [])
        for annotation in annotation_list
    }

    combined_data = {}
    for value_entry in property_values:
        annotation_id = value_entry['annotationId']
        values = value_entry['values']
        flat_values = {}
        for prop_id, prop_value in values.items():
            prop_name = property_id_to_name.get(prop_id, prop_id)
            if isinstance(prop_value, dict):
                flattened = flatten_properties(prop_value, prop_name)
                flat_values.update(flattened)
            else:
                flat_values[prop_name] = prop_value
        # Include tags from the annotation
        tags = annotation_id_to_tags.get(annotation_id, [])
        flat_values['tags'] = tags
        combined_data[annotation_id] = flat_values
    return combined_data

# Function to create a Pandas DataFrame from the combined data
def create_dataframe(combined_data):
    """
    Converts the combined data into a Pandas DataFrame.

    Args:
        combined_data (dict): The combined property data.

    Returns:
        DataFrame: A Pandas DataFrame with annotations as rows and properties as columns.
    """
    df = pd.DataFrame.from_dict(combined_data, orient='index')
    df.index.name = 'annotationId'
    return df

# Function to normalize property names by creating new merged columns
def create_merged_columns(df, merge_map):
    """
    Creates new merged columns based on the merge mapping.

    Args:
        df (DataFrame): The DataFrame with original property columns.
        merge_map (dict): A mapping from new column names to lists of existing column names to merge.

    Returns:
        DataFrame: The DataFrame with new merged columns added.
    """
    for new_col, old_cols in merge_map.items():
        # Create the new column by taking the first non-null value from the old columns
        df[new_col] = df[old_cols].bfill(axis=1).iloc[:, 0]
    return df

# Function to handle missing properties gracefully
def handle_missing_properties(df):
    """
    Ensures consistent data types and handles missing values appropriately.

    Args:
        df (DataFrame): The DataFrame to process.

    Returns:
        DataFrame: The processed DataFrame.
    """
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

# Function to get properties available for specific tags
def get_annotations_by_tag(df, tag):
    """
    Retrieves annotations that contain a specific tag.

    Args:
        df (DataFrame): The DataFrame containing annotation data.
        tag (str): The tag to filter annotations by.

    Returns:
        DataFrame: A filtered DataFrame containing only annotations with the specified tag.
    """
    return df[df['tags'].apply(lambda tags: tag in tags if tags else False)]

def get_index_by_tag(df, tag):
    """
    Retrieves the index of annotations that contain a specific tag.

    Args:
        df (DataFrame): The DataFrame containing annotation data.
        tag (str): The tag to filter annotations by.

    Returns:
        Index: The index of the annotations with the specified tag.
    """
    return df['tags'].apply(lambda tags: tag in tags if tags else False)

def generate_property_values(df, columns, datasetId, propertyId):
    """
    Generates a list of property value dictionaries from the DataFrame.

    Args:
        df (DataFrame): The DataFrame containing annotation data.
        columns (list): A list of column names to include in the property values.
        datasetId (str): The dataset ID to include in each property value.
        propertyId (str): The property ID to use as the primary key in the values dictionary.

    Returns:
        list: A list of property value dictionaries with the specified structure.
    """
    property_values = []
    for annotationId, row in df.iterrows():
        # Initialize the values dictionary for the current annotation
        values_dict = {}
        for col in columns:
            if col in row and pd.notna(row[col]):
                values_dict[col] = row[col]
            else:
                # If the value is missing or NaN, set value to None
                values_dict[col] = None

        # Only proceed if there is at least one valid value
        if values_dict:
            property_value_entry = {
                'annotationId': annotationId,
                'datasetId': datasetId,
                'values': {
                    propertyId: values_dict
                }
            }
            property_values.append(property_value_entry)
        else:
            # Optionally, you can choose to include annotations without any valid values
            # For now, we'll skip annotations with no valid values
            continue

    return property_values
