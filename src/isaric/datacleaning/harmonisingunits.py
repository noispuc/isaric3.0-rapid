from .utils import convert_units

def harmonise_units(df, unit_map):
    """
    Description:
        Harmonises measurement units across columns using a predefined mapping.

    Args:
        df (pandas.DataFrame): Input dataset.
        unit_map (dict): Dictionary mapping column names to target units. 
                         Example: {'weight': 'kg', 'height': 'cm'}

    Returns:
        pandas.DataFrame: Dataset with harmonised units.
    """
    return convert_units(df, unit_map)
