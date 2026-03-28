import sys
import pandas as pd
import numpy as np
from vehicle_insurance.configurations.mongodb_connection import MongoDBConnection
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.constants import DATABASE_NAME

logger = setup_logger()


class InsuranceData(MongoDBConnection):
    def __init__(self, database_name=DATABASE_NAME):
        super().__init__(database_name)

    def get_collection(self, collection_name, database_name=None):
        try:
            if database_name:
                db = self.client[database_name]
            else:
                db = self.database
            collection = db[collection_name]
            logger.info(
                f"Accessed collection: {collection_name} successfully.Now fetching data."
            )

            cursor = collection.find({}, {"_id": 0})

            df = pd.DataFrame(list(cursor))
            logger.info(
                f"Data fetched successfully from collection: {collection_name} and total records fetched: {len(df)}"
            )
            if df.empty:
                logger.warning(f"No data found in collection: {collection_name}")
                return df
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
