def retrieve_attributes(dataclass):
    """
    This functions in the same way as `dataclasses.asdict`, except that the
    elements of the returned dict are identically the same as those of the
    dataclass.
    `asdict` calls `copy.deepcopy` on every element.

    TODO: accept `dict_factory` argument, like `asdict`.
    """
    return {name: getattr(dataclass, name)
            for name in dataclass.__dataclass_fields__}

def prune_attr_dict(dataclass, attr_dict: dict):
    """Given a set of attributes, return the subset which applies
    to this kernel.
    """
    class_fields = dataclass.__dataclass_fields__.keys()
    return {name: value for name,value in attr_dict.items()
            if name in class_fields}
