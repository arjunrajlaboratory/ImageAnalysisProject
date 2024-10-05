import pandas as pd

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
def flatten_properties(properties, parent_key='', delimiter='.'):
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
        new_key = f"{parent_key}{delimiter}{key}" if parent_key else key
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

def create_dataframe_from_annotations(property_values, property_id_to_name, annotation_list):
    """
    Creates a DataFrame directly from annotations and property values.

    Args:
        property_values (list): A list of property value dictionaries.
        property_id_to_name (dict): A mapping from property IDs to names.
        annotation_list (list): A list of annotation dictionaries.

    Returns:
        DataFrame: A Pandas DataFrame with annotations as rows and properties as columns.
    """
    # Create a DataFrame with all annotations
    df = pd.DataFrame(index=[annotation['_id'] for annotation in annotation_list])
    df.index.name = 'annotationId'

    # Add tags column
    df['tags'] = [annotation.get('tags', []) for annotation in annotation_list]

    # Create a dictionary to store flattened property names
    flattened_props = set()

    # Process property values
    for value_entry in property_values:
        annotation_id = value_entry['annotationId']
        values = value_entry['values']
        
        for prop_id, prop_value in values.items():
            prop_name = property_id_to_name.get(prop_id, prop_id)
            
            if isinstance(prop_value, dict):
                flattened = flatten_properties(prop_value, prop_name)
                for flat_name, flat_value in flattened.items():
                    df.at[annotation_id, flat_name] = flat_value
                    flattened_props.add(flat_name)
            else:
                df.at[annotation_id, prop_name] = prop_value
                flattened_props.add(prop_name)

    # Ensure all columns exist in the DataFrame
    for prop in flattened_props:
        if prop not in df.columns:
            df[prop] = None

    return df

# Function to convert given columns from the dataframe into a list of property values
def convert_columns_to_property_values(df, datasetId, propertyId, columns=None):
    """
    Converts specified columns from the DataFrame into a list of property value dictionaries.

    Args:
        df (DataFrame): The DataFrame containing the data.
        datasetId (str): The ID of the dataset.
        propertyId (str): The ID of the property.
        columns (list, optional): A list of column names to convert. If None, use all columns.

    Returns:
        list: A list of property value dictionaries.
    """
    # If columns is None, use all columns from the DataFrame
    if columns is None:
        columns = df.columns.tolist()
    
    # Create a new DataFrame with only the specified columns
    df_subset = df[columns]

    # Initialize the list to store property values
    property_values = []

    # Iterate through the DataFrame rows
    for annotationId, row in df_subset.iterrows():
        # Create a dictionary for the current row, replacing NaN with None
        row_dict = row.where(pd.notna(row), None).to_dict()

        # Create the property value dictionary
        property_value = {
            'annotationId': annotationId,
            'datasetId': datasetId,
            'values': {
                propertyId: row_dict
            }
        }

        property_values.append(property_value)

    return property_values

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

def get_property_info(annotation_client, property_value_list):
    property_info = []
    property_ids = set()

    def get_value_type(value):
        if isinstance(value, dict):
            return {k: get_value_type(v) for k, v in value.items()}
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, str):
            return "string"
        else:
            return "unknown"

    # First pass: collect property IDs and determine value types
    value_types = {}
    for item in property_value_list:
        if 'values' in item:
            for prop_id, value in item['values'].items():
                property_ids.add(prop_id)
                if prop_id not in value_types:
                    value_types[prop_id] = get_value_type(value)

    # Second pass: fetch property details and compile final list
    for prop_id in property_ids:
        prop = annotation_client.getPropertyById(prop_id)
        
        detail = {
            "_id": prop["_id"],
            "name": prop["name"],
            "image": prop.get("image", ""),
            "tags": prop.get("tags", {}).get("tags", []),
            "shape": prop.get("shape", ""),
            "value": value_types.get(prop_id, "not found")
        }
        
        property_info.append(detail)

    return property_info