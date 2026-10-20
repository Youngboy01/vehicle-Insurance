import pandas as pd
import pymongo
import pytest

from vehicle_insurance.data_access.Insurance_data import InsuranceData


class _FakeCursor:
    def __init__(self, records):
        self.records = records
        self.closed = False

    def __iter__(self):
        return iter(self.records)

    def sort(self, sort_spec):
        assert sort_spec == [("id", pymongo.ASCENDING)]
        self.records = sorted(self.records, key=lambda record: record["id"])
        return self

    def hint(self, index_name):
        assert index_name == "id_1"
        return self

    def close(self):
        self.closed = True


class _FakeCollection:
    def __init__(self, records):
        self.records = records
        self.batch_size = None
        self.cursor = None

    def index_information(self):
        return {"id_1": {"key": [("id", pymongo.ASCENDING)]}}

    def find(self, query, projection, *, batch_size, session):
        assert query == {}
        assert projection["_id"] == 0
        assert projection["id"] == 1
        assert projection["Response"] == 1
        assert "marker" not in projection
        self.batch_size = batch_size
        self.cursor = _FakeCursor(
            [
                {
                    field: document[field]
                    for field, included in projection.items()
                    if included and field != "_id" and field in document
                }
                for document in self.records
            ]
        )
        return self.cursor


class _FakeDatabase:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, collection_name):
        assert collection_name == "INSURANCE_DATA"
        return self.collection

    name = "INSURANCE"


class _FakeSession:
    cluster_time = {"clusterTime": type("Timestamp", (), {"time": 10, "inc": 2})()}
    operation_time = type("Timestamp", (), {"time": 10, "inc": 3})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class _FakeClient:
    def __init__(self, database):
        self.database = database

    def __getitem__(self, database_name):
        assert database_name == "INSURANCE"
        return self.database

    def start_session(self, *, snapshot):
        assert snapshot is True
        return _FakeSession()


def test_get_collection_uses_batched_cursor_and_closes_it():
    records = [
        {"id": 2, "Response": 1, "marker": "na"},
        {"id": 1, "Response": 0},
    ]
    collection = _FakeCollection(records)
    insurance_data = object.__new__(InsuranceData)
    database = _FakeDatabase(collection)
    insurance_data.database = database
    insurance_data.client = _FakeClient(database)

    result = insurance_data.get_collection("INSURANCE_DATA")

    assert isinstance(result, pd.DataFrame)
    assert result["id"].tolist() == [1, 2]
    assert "marker" not in result.columns
    assert collection.batch_size == 10_000
    assert collection.cursor.closed is True


def test_snapshot_manifest_records_read_boundary_and_source_contract():
    records = [
        {"id": 1, "Response": 0},
        {"id": 2, "Response": 1},
    ]
    collection = _FakeCollection(records)
    insurance_data = object.__new__(InsuranceData)
    database = _FakeDatabase(collection)
    insurance_data.database = database
    insurance_data.client = _FakeClient(database)

    result = insurance_data.extract_snapshot("INSURANCE_DATA")

    assert result.manifest["snapshot_id"].startswith("mongodb-sha256:")
    assert result.manifest["source"]["namespace"] == "INSURANCE.INSURANCE_DATA"
    assert result.manifest["read_boundary"]["mode"] == "snapshot_session"
    assert result.manifest["read_boundary"]["cluster_time"] == {
        "seconds": 10,
        "increment": 2,
    }
    assert result.manifest["projection"]["mongo_projection"]["_id"] == 0
    assert result.manifest["ordering"]["index_name"] == "id_1"
    assert result.manifest["row_count"] == 2
    assert result.manifest["id_range"] == {"field": "id", "min": 1, "max": 2}


def test_snapshot_retryable_read_failure_aborts_without_retry():
    class FailingCursor(_FakeCursor):
        def __iter__(self):
            raise pymongo.errors.AutoReconnect("cursor interrupted")

    class FailingCollection(_FakeCollection):
        def find(self, query, projection, *, batch_size, session):
            self.cursor = FailingCursor(self.records)
            return self.cursor

    collection = FailingCollection([{"id": 1, "Response": 0}])
    insurance_data = object.__new__(InsuranceData)
    database = _FakeDatabase(collection)
    insurance_data.database = database
    insurance_data.client = _FakeClient(database)

    with pytest.raises(Exception, match="aborted without retrying"):
        insurance_data.extract_snapshot("INSURANCE_DATA")
