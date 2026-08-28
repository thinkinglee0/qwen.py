import os


def _get_evn_list_value(env_name, default_value) -> list[int]:
    env = os.environ.get(key=env_name)
    if env is None:
        return default_value
    return [int(b) for b in env.split(",")]