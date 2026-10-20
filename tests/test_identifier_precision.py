import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import app
from vehicle_insurance.constants import SCHEMA_FILE_PATH
from vehicle_insurance.contracts.data_contract import DataContractValidator
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException

MAX_CUSTOMER_ID = 2**53 - 1


def _scoring_record(customer_id: int) -> dict:
    return {
        "id": customer_id,
        "Gender": "Male",
        "Age": 35,
        "Driving_License": 1,
        "Region_Code": 28.0,
        "Previously_Insured": 0,
        "Vehicle_Age": "1-2 Year",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 25000.0,
        "Policy_Sales_Channel": 1.0,
        "Vintage": 5,
    }


def test_shared_contract_enforces_browser_safe_id_range() -> None:
    validator = DataContractValidator(SCHEMA_FILE_PATH)
    valid = pd.DataFrame([_scoring_record(1), _scoring_record(MAX_CUSTOMER_ID)])

    validated = validator.validate(valid, mode="scoring")

    assert validated["id"].tolist() == [1, MAX_CUSTOMER_ID]

    invalid = valid.copy()
    invalid.loc[1, "id"] = MAX_CUSTOMER_ID + 1
    with pytest.raises(VehicleInsuranceException):
        validator.validate(invalid, mode="scoring")


@pytest.mark.parametrize("customer_id", [0, -1, MAX_CUSTOMER_ID + 1])
def test_api_rejects_out_of_range_ids(customer_id: int) -> None:
    with pytest.raises(ValidationError):
        app.PredictRequest(**_scoring_record(customer_id))


def test_rank_preserves_boundary_ids_and_numeric_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_engine = SimpleNamespace(
        model_id="model-id",
        model_uri="models:/model-id",
        source_run_id="run-id",
        model_version="model-id",
        operating_threshold=None,
    )
    monkeypatch.setattr(app, "current_engine", fake_engine)
    monkeypatch.setattr(
        app,
        "_score_raw_dataframe",
        lambda _engine, dataframe: np.array([0.9, 0.9, 0.8]),
    )
    request = app.RankRequest(
        records=[
            app.PredictRequest(**_scoring_record(MAX_CUSTOMER_ID)),
            app.PredictRequest(**_scoring_record(MAX_CUSTOMER_ID - 1)),
            app.PredictRequest(**_scoring_record(38113)),
        ],
        top_k_count=3,
    )

    response = asyncio.run(app.rank(request))

    assert [customer.id for customer in response.selected_customers] == [
        MAX_CUSTOMER_ID - 1,
        MAX_CUSTOMER_ID,
        38113,
    ]


def test_dashboard_id_parser_rejects_unsafe_and_permissive_forms() -> None:
    template = Path("templates/index.html").read_text(encoding="utf-8")
    start = template.index("  const MAX_CUSTOMER_ID = Number.MAX_SAFE_INTEGER;")
    end = template.index("\n\n  function setServiceStatus", start)
    helper = template[start:end]
    node_script = f"""
{helper}
const valid = [parseCustomerId('1'), parseCustomerId('9007199254740991')];
const invalid = ['', '1.5', '1e3', '123abc', '9007199254740992',
  '9007199254740991.9'].map((value) => {{
  try {{ parseCustomerId(value); return false; }}
  catch (_) {{ return true; }}
}});
const invalidBatch = [
  [{{id: 0}}],
  [{{id: 9007199254740992}}],
  [{{id: 1.5}}],
  [{{id: '1'}}],
  [{{id: 9007199254740991}}],
].map((records) => {{
  try {{ validateBatchCustomerIds(records); return false; }}
  catch (_) {{ return true; }}
}});
const rawFractional = '[{{"id":9007199254740991.1}}]';
let rawFractionalRejected = false;
try {{ validateRawBatchCustomerIds(rawFractional); }}
catch (_) {{ rawFractionalRejected = true; }}
const rawBoundary = '[{{"id":9007199254740991}}]';
validateRawBatchCustomerIds(rawBoundary);
console.log(JSON.stringify({{valid, invalid, invalidBatch}}));
console.log(JSON.stringify({{rawFractionalRejected}}));
"""
    completed = subprocess.run(
        ["node", "-e", node_script],
        check=True,
        capture_output=True,
        text=True,
    )
    result_lines = completed.stdout.splitlines()
    result = json.loads(result_lines[0])
    raw_result = json.loads(result_lines[1])

    assert result == {
        "valid": [1, MAX_CUSTOMER_ID],
        "invalid": [True, True, True, True, True, True],
        "invalidBatch": [True, True, True, True, False],
    }
    assert raw_result == {"rawFractionalRejected": True}
