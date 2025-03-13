import os
import json

DEFAULT_ATTRIBUTES = ["mudablue", "crosssim", "repopal","mock1"]

def load_integrate_level_low():
    return load_config_value("integrate_level_low", default_value=1, expected_type=int)

def load_integrate_level_high():
    return load_config_value("integrate_level_high", default_value=4, expected_type=int)
def load_pick_repo_number():
    value = load_config_value("pick_repo_num", default_value=300, expected_type=int)
    if value==0:
        return None
    return load_config_value("pick_repo_num", default_value=300, expected_type=int)
def load_num_steps():
    return load_config_value("t1_range_steps", default_value=20, expected_type=int)

def load_normalize():
    return load_config_value("normalize", default_value=True, expected_type=bool)

def load_attributes():
    return load_config_value("custom_attributes", default_value=DEFAULT_ATTRIBUTES, expected_type=list)

def load_config_value(key, default_value, expected_type=None):
    config_dic = load_config_json()
    if not config_dic:
        print(f"Config is empty, [Using default {key}]: {default_value}")
        return default_value

    value = config_dic.get(key)
    if value is None:
        print(f"{key} not found in config, [Using default {key}]: {default_value}")
        return default_value

    if expected_type and not isinstance(value, expected_type):
        print(f"Invalid format for {key} in config, [Using default {key}]: {default_value}")
        return default_value

    print(f"{key} loaded from config file: {value}")
    return value



def load_config_json(config_path="config.json"):
    """
    Load JSON configuration file.
    If the file does not exist or contains errors, return an empty dictionary.
    """
    if not os.path.exists(config_path):
        # If not in current directory, check parent directory
        parent_path = os.path.join("..", config_path)
        if os.path.exists(parent_path):
            config_path = parent_path
        else:
            print(f"The config file '{config_path}' does not exist in the current or parent directory")
            return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to load the config file: {e}")
        return {}


load_attributes()
load_normalize()
load_integrate_level_high()
load_integrate_level_low()
load_num_steps()
load_pick_repo_number()