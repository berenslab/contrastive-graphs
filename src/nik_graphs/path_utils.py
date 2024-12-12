def try_convert(val, convert):
    try:
        return convert(val)
    except ValueError:
        return val


def path_to_kwargs(path):
    parts = path.name.split(",")
    name = parts.pop(0)
    kwargs = {}
    for p in parts:
        key, val = p.split("=")

        # try to convert to either an int or float, otherwise pass the
        # value on as is.
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        # special cases for val can be handled here before being
        # passed on to the kwarg dict.

        kwargs[key] = val

    return name, kwargs
