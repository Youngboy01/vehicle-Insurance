import os
import yaml
import sys
import pickle
import numpy as np
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()


def read_yaml_file(file_path: str):
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def save_object(file_path: str, obj):
    try:
        logger.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def load_object(file_path: str):
    try:
        logger.info(f"Loading object from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        logger.info(f"Saving numpy array to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy array saved successfully to {file_path}")
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        logger.info(f"Loading numpy array from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj, allow_pickle=False)
        logger.info(f"Numpy array loaded successfully from {file_path}")
        return array
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e
