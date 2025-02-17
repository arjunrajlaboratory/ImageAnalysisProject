def convert_units(pixelSize, to_units):
    """
    Convert a pixel size to a different unit.
    """
    current_units = pixelSize['unit']
    current_value = pixelSize['value']
    if current_units == 'm':
        current_value *= 1
    elif current_units == 'mm':
        current_value *= 1e-3
    elif current_units == 'µm':
        current_value *= 1e-6
    elif current_units == 'nm':
        current_value *= 1e-9
    else:
        raise ValueError(f"Unknown unit: {current_units}")

    if to_units == 'm':
        current_value *= 1
    elif to_units == 'mm':
        current_value *= 1e3
    elif to_units == 'µm':
        current_value *= 1e6
    elif to_units == 'nm':
        current_value *= 1e9
    else:
        raise ValueError(f"Unknown unit: {to_units}")

    return {'unit': to_units, 'value': current_value}
