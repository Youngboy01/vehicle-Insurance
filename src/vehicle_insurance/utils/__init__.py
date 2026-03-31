import yaml
import sys
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException

def read_yaml_file(file_path:str):
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e