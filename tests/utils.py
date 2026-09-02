import os


def parse_env_list_value(env_name, default_value) -> list[int] | list[str] | list[bool]:
    env = os.environ.get(key=env_name)
    if env is None:
        return default_value

    values = [item.strip() for item in env.split(",") if item.strip()]
    if not values:
        return default_value

    if not default_value:
        first = values[0].lower()
        if first in {"true", "false"}:
            return [item.lower() == "true" for item in values]
        try:
            return [int(item) for item in values]
        except ValueError:
            return values

    item_type = type(default_value[0])
    if item_type is bool:
        true_values = {"1", "true", "yes", "on"}
        return [item.lower() in true_values for item in values]
    if item_type is int:
        return [int(item) for item in values]
    if item_type is str:
        return [str(item) for item in values]

    return [item_type(item) for item in values]