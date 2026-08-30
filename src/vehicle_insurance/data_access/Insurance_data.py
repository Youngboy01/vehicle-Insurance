import sys
import time

import numpy as np
import pandas as pd
import pymongo

from vehicle_insurance.configurations.mongodb_connection import MongoDBConnection
from vehicle_insurance.constants import DATABASE_NAME
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()

FETCH_BATCH_SIZE = 10_000
FETCH_MAX_ATTEMPTS = 3
FETCH_RETRY_DELAY_SECONDS = 2

_RETRYABLE_MONGO_ERRORS = tuple(
    error_type
    for error_type in (
        pymongo.errors.AutoReconnect,
        pymongo.errors.NetworkTimeout,
        pymongo.errors.ServerSelectionTimeoutError,
        getattr(pymongo.errors, "_OperationCancelled", None),
    )
    if error_type is not None
)


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

            records = None
            for attempt in range(1, FETCH_MAX_ATTEMPTS + 1):
                cursor = None
                try:
                    cursor = collection.find(
                        {},
                        {"_id": 0},
                        batch_size=FETCH_BATCH_SIZE,
                    )
                    records = []
                    for record_number, document in enumerate(cursor, start=1):
                        records.append(document)
                        if record_number % FETCH_BATCH_SIZE == 0:
                            logger.info(
                                "MongoDB extraction progress: fetched %d records",
                                record_number,
                            )
                    break
                except _RETRYABLE_MONGO_ERRORS as error:
                    if attempt == FETCH_MAX_ATTEMPTS:
                        raise
                    delay = FETCH_RETRY_DELAY_SECONDS * attempt
                    logger.warning(
                        "MongoDB cursor interrupted on attempt %d/%d: %s. "
                        "Retrying in %d seconds.",
                        attempt,
                        FETCH_MAX_ATTEMPTS,
                        error,
                        delay,
                    )
                    time.sleep(delay)
                finally:
                    if cursor is not None:
                        cursor.close()

            if records is None:
                raise RuntimeError("MongoDB extraction returned no records")

            df = pd.DataFrame.from_records(records)
            # Do not request a server-side sort here.  The benchmark
            # collection has no guaranteed supporting index, so MongoDB can
            # reject a large in-memory sort at its 32 MB limit.  Sorting the
            # materialized extraction locally gives deterministic notebook-
            # compatible ordering without mutating the database schema.
            if not df.empty and "id" in df.columns:
                df.sort_values("id", kind="mergesort", inplace=True)
                df.reset_index(drop=True, inplace=True)
            logger.info(
                f"Data fetched successfully from collection: {collection_name} and total records fetched: {len(df)}"
            )
            if df.empty:
                logger.warning(f"No data found in collection: {collection_name}")
                return df
            df = df.replace({"na": np.nan}).infer_objects(copy=False)
            return df
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
