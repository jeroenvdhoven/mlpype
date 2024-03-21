from typing import Any, Dict


def get_args_for_prefix(prefix: str, dct: Dict[str, Any]) -> Dict[str, Any]:
    """Get all keys starting with `prefix` and remove the prefix.

    Args:
        prefix (str): The prefix to search. In the resulting dictionary these prefixes will
            be removed from the keys.
        dct (Dict[str, Any]): The dictionary to search for the given prefix.

    Returns:
        Dict[str, Any]: dct, reduced by only including keys that included the prefix and where
            that prefix is removed from the original key.
    """
    return {key.replace(prefix, "", 1): value for key, value in dct.items() if key.startswith(prefix)}
