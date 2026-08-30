import os
import sys

import certifi
import pymongo
from dotenv import load_dotenv

from vehicle_insurance.constants.__init__ import DATABASE_NAME, MONGODB_URL_KEY
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

load_dotenv()

certi = certifi.where()
logger = setup_logger()


class MongoDBConnection:
    client = None

    def __init__(self, database_name=DATABASE_NAME):
        try:
            # check for mongodb connection
            if MongoDBConnection.client is None:
                mongodb_url = os.getenv(MONGODB_URL_KEY)
                if not mongodb_url:
                    raise Exception(
                        f"{MONGODB_URL_KEY} environment variable is not set."
                    )

                # establish connection to mongodb
                MongoDBConnection.client = pymongo.MongoClient(
                    mongodb_url,
                    tlsCAFile=certi,
                    connectTimeoutMS=30_000,
                    serverSelectionTimeoutMS=30_000,
                    socketTimeoutMS=180_000,
                    # A full benchmark extraction can exceed a short
                    # client-side operation deadline. Each socket response
                    # remains bounded by socketTimeoutMS.
                    timeoutMS=None,
                )

            self.client = MongoDBConnection.client
            self.database = self.client[database_name]
            logger.info(f"Connected to MongoDB database: {database_name}")
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
