import re
import tomli

import lazyConfig


def get_config(file):
    with open(file, "rb") as f:
        config = tomli.load(f)
    refine_nodes(config, config)
    config = lazyConfig.from_primitive(config)
    return config


def refine_nodes(node, config):
    for k, v in node.items():
        if isinstance(v, str):
            replace_str(k, node, config)
        elif isinstance(v, dict):
            refine_nodes(v, config)
        elif isinstance(v, list):
            for i, inner_v in enumerate(v):
                if isinstance(inner_v, str):
                    replace_str(k, node, config, list_index=i)
                elif isinstance(inner_v, dict):
                    refine_nodes(inner_v, config)


def replace_str(key, node, config, list_index=None):
    """
    If `node[key]` is list type, set list_index to the index of current element of `node[key]`.

    Args:
        key (str):
        node (mapping):
        config (mapping):
        list_index (int or None, optional, default=None):

    Returns:

    """

    reference_patten = r"(\$\{.+?\})"
    reference_key_patten = r"\$\{(.+)\}"
    if list_index is not None:
        value = node[key][list_index]
    else:
        value = node[key]
    match = re.findall(reference_patten, value)
    if not match:
        return
    for each in match:
        replace_key = re.search(reference_key_patten, each).group(1)
        replace_node, replace_key = get_node_key(replace_key, config)
        if isinstance(replace_node[replace_key], str):
            replace_str(replace_key, replace_node, config)
        if list_index is not None:
            node[key][list_index] = node[key][list_index].replace(each, config[replace_key])
        else:
            node[key] = node[key].replace(each, str(replace_node[replace_key]))


def get_node_key(key, config):
    """
    Get the inner node and key.

    Args:
        key (str):
        config (mapping):

    Returns:

    """

    node_hierarchy = key.split(".")

    if len(node_hierarchy) == 1:
        return config, key
    node = config
    for key in node_hierarchy[:-1]:
        node = node[key]
    return node, node_hierarchy[-1]
