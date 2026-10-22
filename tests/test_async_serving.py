import asyncio
import json
import threading
from types import SimpleNamespace

import anyio
import pytest

import app


def _record(customer_id: int = 1) -> app.PredictRequest:
    return app.PredictRequest(
        id=customer_id,
        Gender="Male",
        Age=35,
        Driving_License=1,
        Region_Code=28.0,
        Previously_Insured=0,
        Vehicle_Age="1-2 Year",
        Vehicle_Damage="Yes",
        Annual_Premium=25000.0,
        Policy_Sales_Channel=1.0,
        Vintage=5,
    )


def _rank_request(size: int = 2) -> app.RankRequest:
    return app.RankRequest(
        records=[_record(customer_id) for customer_id in range(1, size + 1)],
        top_k_count=1,
    )


def test_oversized_http_body_is_rejected_before_json_parsing() -> None:
    body = b"{" + b"x" * app.MAX_REQUEST_BODY_BYTES + b"}"

    status, response_body = asyncio.run(
        _asgi_request("/predict", method="POST", body=body)
    )

    assert status == 413
    assert json.loads(response_body)["error_code"] == "REQUEST_BODY_TOO_LARGE"


def test_scoring_string_length_is_bounded_before_model_work() -> None:
    record = _record().model_dump()
    record["Gender"] = "M" * 65
    body = json.dumps(record).encode("utf-8")

    status, response_body = asyncio.run(
        _asgi_request("/predict", method="POST", body=body)
    )

    response = json.loads(response_body)
    assert status == 422
    assert response["error_code"] == "SCORING_SCHEMA_INVALID"
    assert "Gender" in response["fields"]


def test_excessive_batch_is_rejected_before_scoring() -> None:
    body = json.dumps(
        {
            "records": [
                _record(customer_id).model_dump()
                for customer_id in range(1, 10_002)
            ],
            "top_k_count": 1,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(body) < app.MAX_REQUEST_BODY_BYTES

    status, response_body = asyncio.run(
        _asgi_request("/rank", method="POST", body=body)
    )

    assert status == 422
    assert json.loads(response_body)["error_code"] == "SCORING_SCHEMA_INVALID"


def test_http_concurrent_scoring_request_is_rejected_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_rank(engine, request):
        started.set()
        assert release.wait(timeout=5)
        return app.RankResponse(
            selected_customers=[],
            total_scored=1,
            selected_count=1,
            capacity_type="top_k_count",
            capacity_value=1,
            model_id=engine.model_id,
            model_uri=engine.model_uri,
            source_run_id=engine.source_run_id,
            model_version=engine.model_version,
            operating_threshold=None,
            api_timestamp="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(app, "current_engine", _engine("model"))
    monkeypatch.setattr(app, "SCORING_OPERATION_LIMITER", anyio.CapacityLimiter(1))
    monkeypatch.setattr(app, "_build_rank_response", slow_rank)
    body = json.dumps(
        {"records": [_record().model_dump()], "top_k_count": 1}
    ).encode("utf-8")

    async def scenario():
        first_task = asyncio.create_task(
            _asgi_request("/rank", method="POST", body=body)
        )
        assert await asyncio.to_thread(started.wait, 5)
        busy_status, busy_body = await _asgi_request(
            "/rank", method="POST", body=body
        )
        release.set()
        first_status, first_body = await first_task
        return first_status, first_body, busy_status, busy_body

    first_status, first_body, busy_status, busy_body = asyncio.run(scenario())

    assert first_status == 200
    assert json.loads(first_body)["total_scored"] == 1
    assert busy_status == 503
    assert json.loads(busy_body)["error_code"] == "SCORING_BUSY"


def test_untrusted_origin_preflight_is_not_granted_cors_access() -> None:
    response_headers = []
    status, _body = asyncio.run(
        _asgi_request(
            "/predict",
            method="OPTIONS",
            headers=(
                (b"origin", b"https://untrusted.example"),
                (b"access-control-request-method", b"POST"),
            ),
            response_headers=response_headers,
        )
    )

    assert status == 405
    assert all(
        name.lower() != b"access-control-allow-origin"
        for name, _value in response_headers
    )


def test_reload_requires_server_side_token_and_accepts_only_the_exact_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement_engine = _engine("replacement-model")
    monkeypatch.setenv("RELOAD_TOKEN", "synthetic-token")
    monkeypatch.setattr(app, "current_engine", _engine("old-model"))
    monkeypatch.setattr(
        app,
        "RELOAD_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )
    monkeypatch.setattr(
        app,
        "_load_engine_from_mlflow",
        lambda: replacement_engine,
    )

    missing_status, missing_body = asyncio.run(
        _asgi_request("/reload", method="POST")
    )
    wrong_status, wrong_body = asyncio.run(
        _asgi_request(
            "/reload",
            method="POST",
            headers=((b"x-reload-token", b"wrong-token"),),
        )
    )
    correct_status, correct_body = asyncio.run(
        _asgi_request(
            "/reload",
            method="POST",
            headers=((b"x-reload-token", b"synthetic-token"),),
        )
    )

    assert missing_status == 422
    assert json.loads(missing_body)["error_code"] == "REQUEST_SCHEMA_INVALID"
    assert wrong_status == 401
    assert json.loads(wrong_body)["error_code"] == "AUTHENTICATION_FAILED"
    assert correct_status == 200
    assert json.loads(correct_body)["model_id"] == replacement_engine.model_id
    assert app.current_engine is replacement_engine


async def _asgi_request(
    path: str,
    *,
    method: str = "GET",
    body: bytes = b"",
    headers: tuple[tuple[bytes, bytes], ...] = (),
    response_headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, bytes]:
    """Exercise the FastAPI app boundary without starting a real server."""

    sent_body = False
    status_code = 0
    response_chunks: list[bytes] = []

    async def receive():
        nonlocal sent_body
        if sent_body:
            return {"type": "http.disconnect"}
        sent_body = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        nonlocal status_code
        if message["type"] == "http.response.start":
            status_code = message["status"]
            if response_headers is not None:
                response_headers.extend(message.get("headers", ()))
        elif message["type"] == "http.response.body":
            response_chunks.append(message.get("body", b""))

    encoded_path = path.encode("ascii")
    await app.app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": encoded_path,
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                *headers,
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "root_path": "",
        },
        receive,
        send,
    )
    return status_code, b"".join(response_chunks)


def _engine(model_id: str) -> app.PredictionEngine:
    reference = app.ImmutableModelReference(
        model_id=model_id,
        model_uri=f"models:/{model_id}",
        run_id=f"run-{model_id}",
    )
    return app.PredictionEngine(
        model=SimpleNamespace(),
        model_reference=reference,
        evaluation_metrics={"validation": {}, "test": {}},
        lineage={
            "model_spec_version": "3",
            "contract_version": "1.0.0",
            "git_commit": "commit",
            "dataset_snapshot_id": "snapshot",
        },
        raw_feature_names=(),
        operating_threshold=None,
    )


def test_http_scoring_validates_and_ranks_model_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine("model")
    engine.raw_feature_names = app.EXPECTED_RAW_FEATURES

    def predict_proba(features):
        scores = [0.81234 if age == 35 else 0.6 for age in features["Age"]]
        return [[1 - score, score] for score in scores]

    engine.model = SimpleNamespace(predict_proba=predict_proba)
    monkeypatch.setattr(app, "current_engine", engine)
    monkeypatch.setattr(
        app,
        "SCORING_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )

    predict_status, predict_body = asyncio.run(
        _asgi_request(
            "/predict",
            method="POST",
            body=json.dumps(_record().model_dump()).encode("utf-8"),
        )
    )
    prediction = json.loads(predict_body)
    assert predict_status == 200
    assert prediction["response_propensity_score"] == 0.8123
    assert prediction["probability"] == 0.8123
    assert prediction["model_version"] == engine.model_id

    records = [
        _record(2).model_dump(),
        _record(1).model_dump(),
        _record(3).model_dump(),
    ]
    records[-1]["Age"] = 40
    rank_status, rank_body = asyncio.run(
        _asgi_request(
            "/rank",
            method="POST",
            body=json.dumps({"records": records, "top_k_count": 2}).encode("utf-8"),
        )
    )
    ranking = json.loads(rank_body)
    assert rank_status == 200
    assert ranking["total_scored"] == 3
    assert ranking["selected_count"] == 2
    assert [customer["id"] for customer in ranking["selected_customers"]] == [1, 2]
    assert [
        customer["response_propensity_score"]
        for customer in ranking["selected_customers"]
    ] == [0.8123, 0.8123]
    assert ranking["model_id"] == engine.model_id
    assert ranking["source_run_id"] == engine.source_run_id

    invalid_record = _record().model_dump()
    invalid_record["Age"] = 17
    invalid_status, invalid_body = asyncio.run(
        _asgi_request(
            "/predict",
            method="POST",
            body=json.dumps(invalid_record).encode("utf-8"),
        )
    )
    assert invalid_status == 422
    invalid_response = json.loads(invalid_body)
    assert invalid_response["error_code"] == "SCORING_VALUE_INVALID"
    assert "Age" in invalid_response["fields"]

    engine.model = SimpleNamespace(predict_proba=lambda _features: [[0.81234]])
    failure_status, failure_body = asyncio.run(
        _asgi_request(
            "/predict",
            method="POST",
            body=json.dumps(_record().model_dump()).encode("utf-8"),
        )
    )
    assert failure_status == 500
    assert json.loads(failure_body)["error_code"] == "MODEL_INFERENCE_FAILED"


def test_http_failed_startup_recovers_after_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable():
        raise ConnectionError("synthetic model-store outage")

    monkeypatch.setattr(app, "_load_engine_from_mlflow", unavailable)
    monkeypatch.setattr(app, "current_engine", None)
    monkeypatch.setattr(app, "model_load_error", None)
    monkeypatch.setattr(app, "model_load_failure_code", None)
    monkeypatch.setattr(app, "model_load_attempted_at", None)
    monkeypatch.setattr(app, "RELOAD_OPERATION_LIMITER", anyio.CapacityLimiter(1))
    monkeypatch.setenv("RELOAD_TOKEN", "synthetic-token")
    replacement_engine = _engine("replacement-model")

    async def scenario():
        async with app.lifespan(app.app):
            live_status, live_body = await _asgi_request("/health/live")
            assert live_status == 200
            assert json.loads(live_body) == {"status": "alive"}

            ready_status, ready_body = await _asgi_request("/health/ready")
            assert ready_status == 503
            assert json.loads(ready_body)["error_code"] == "DEPENDENCY_UNAVAILABLE"

            predict_status, predict_body = await _asgi_request(
                "/predict",
                method="POST",
                body=json.dumps(_record().model_dump()).encode("utf-8"),
            )
            assert predict_status == 503
            assert json.loads(predict_body)["error_code"] == "DEPENDENCY_UNAVAILABLE"

            monkeypatch.setattr(app, "_load_engine_from_mlflow", lambda: replacement_engine)
            reload_status, reload_body = await _asgi_request(
                "/reload",
                method="POST",
                headers=((b"x-reload-token", b"synthetic-token"),),
            )
            assert reload_status == 200
            assert json.loads(reload_body)["model_id"] == replacement_engine.model_id

            recovered_status, recovered_body = await _asgi_request("/health/ready")
            assert recovered_status == 200
            recovered = json.loads(recovered_body)
            assert recovered["model_id"] == replacement_engine.model_id
            assert recovered["status"] == "ready"

    asyncio.run(scenario())


def test_rank_offload_keeps_readiness_responsive_and_returns_consistent_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_engine = _engine("old-model")
    replacement_engine = _engine("new-model")
    started = threading.Event()
    release = threading.Event()

    def slow_rank(engine, request):
        started.set()
        assert release.wait(timeout=5)
        return app.RankResponse(
            selected_customers=[],
            total_scored=0,
            selected_count=0,
            capacity_type="top_k_count",
            capacity_value=0,
            model_id=engine.model_id,
            model_uri=engine.model_uri,
            source_run_id=engine.source_run_id,
            model_version=engine.model_version,
            operating_threshold=None,
            api_timestamp="2026-01-01T00:00:00+00:00",
        )

    monkeypatch.setattr(app, "current_engine", old_engine)
    monkeypatch.setattr(
        app,
        "SCORING_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )
    monkeypatch.setattr(app, "_build_rank_response", slow_rank)

    async def scenario():
        body = json.dumps(
            {
                "records": [
                    _record(customer_id).model_dump()
                    for customer_id in range(1, 3)
                ],
                "top_k_count": 1,
            }
        ).encode("utf-8")
        rank_task = asyncio.create_task(
            _asgi_request("/rank", method="POST", body=body)
        )
        assert await asyncio.to_thread(started.wait, 10)

        ready_status, ready_body = await _asgi_request("/health/ready")

        monkeypatch.setattr(app, "current_engine", replacement_engine)
        release.set()
        rank_status, rank_body = await rank_task
        return (
            ready_status,
            json.loads(ready_body),
            rank_status,
            json.loads(rank_body),
        )

    ready_status, ready, rank_status, response = asyncio.run(scenario())

    assert ready_status == 200
    assert ready["status"] == "ready"
    assert rank_status == 200
    assert response["model_id"] == old_engine.model_id


def test_reload_is_offloaded_serialized_and_failed_reload_preserves_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_engine = _engine("old-model")
    replacement_engine = _engine("new-model")
    started = threading.Event()
    release = threading.Event()

    def slow_load():
        started.set()
        assert release.wait(timeout=5)
        return replacement_engine

    monkeypatch.setattr(app, "current_engine", old_engine)
    monkeypatch.setenv("RELOAD_TOKEN", "synthetic-token")
    monkeypatch.setattr(
        app,
        "RELOAD_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )
    monkeypatch.setattr(app, "_load_engine_from_mlflow", slow_load)

    async def successful_reload():
        headers = ((b"x-reload-token", b"synthetic-token"),)
        first_task = asyncio.create_task(
            _asgi_request("/reload", method="POST", headers=headers)
        )
        assert await asyncio.to_thread(started.wait, 5)

        ready_status, ready_body = await _asgi_request("/health/ready")
        ready = json.loads(ready_body)
        assert ready_status == 200
        assert ready["status"] == "ready"
        assert ready["model_id"] == old_engine.model_id

        busy_status, busy_body = await _asgi_request(
            "/reload",
            method="POST",
            headers=headers,
        )
        assert busy_status == 503
        assert json.loads(busy_body)["error_code"] == "RELOAD_BUSY"
        assert app.current_engine is old_engine
        release.set()
        reload_status, reload_body = await first_task
        return reload_status, json.loads(reload_body)

    reload_status, response = asyncio.run(successful_reload())

    assert app.current_engine is replacement_engine
    assert reload_status == 200
    assert response["model_id"] == replacement_engine.model_id

    def failed_load():
        raise ConnectionError("synthetic MLflow outage")

    monkeypatch.setattr(app, "_load_engine_from_mlflow", failed_load)

    failure_status, failure_body = asyncio.run(
        _asgi_request(
            "/reload",
            method="POST",
            headers=((b"x-reload-token", b"synthetic-token"),),
        )
    )

    assert failure_status == 503
    assert json.loads(failure_body)["error_code"] == "DEPENDENCY_UNAVAILABLE"
    assert app.current_engine is replacement_engine


def test_model_load_applies_finite_serving_thread_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreprocessor:
        feature_names_in_ = app.EXPECTED_RAW_FEATURES

        def get_feature_names_out(self):
            return [f"feature_{index}" for index in range(220)]

    class FakeModel:
        classes_ = [0, 1]
        n_features_in_ = 220
        n_jobs = -1

        def set_params(self, **params):
            self.n_jobs = params["n_jobs"]
            return self

        def predict_proba(self, dataframe):
            return [[0.5, 0.5] for _ in range(len(dataframe))]

    reference = app.CURRENT_MODEL_REFERENCE
    pipeline = app.Pipeline(
        steps=[
            ("preprocessor", FakePreprocessor()),
            ("model", FakeModel()),
        ]
    )
    run = SimpleNamespace(
        info=SimpleNamespace(run_id=reference.run_id),
        data=SimpleNamespace(metrics={}, tags={}),
    )
    model_info = SimpleNamespace(
        model_uri=reference.model_uri,
        model_id=reference.model_id,
        run_id=reference.run_id,
    )

    monkeypatch.setattr(
        app,
        "get_mlflow_client",
        lambda: SimpleNamespace(get_run=lambda run_id: run),
    )
    monkeypatch.setattr(app.mlflow.models, "get_model_info", lambda model_uri: model_info)
    monkeypatch.setattr(app.mlflow.sklearn, "load_model", lambda model_uri: pipeline)

    engine = app._load_engine_from_mlflow(reference)

    assert engine.model.named_steps["model"].n_jobs == app.SERVING_MODEL_THREADS


def test_model_load_rejects_mismatched_model_identity_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = app.CURRENT_MODEL_REFERENCE
    downloads = []
    monkeypatch.setattr(
        app,
        "get_mlflow_client",
        lambda: SimpleNamespace(
            get_run=lambda run_id: SimpleNamespace(info=SimpleNamespace(run_id=run_id))
        ),
    )
    monkeypatch.setattr(
        app.mlflow.models,
        "get_model_info",
        lambda model_uri: SimpleNamespace(
            model_uri=model_uri,
            model_id="different-model-id",
            run_id=reference.run_id,
        ),
    )
    monkeypatch.setattr(
        app.mlflow.sklearn,
        "load_model",
        lambda model_uri: downloads.append(model_uri),
    )

    with pytest.raises(
        app.VehicleInsuranceException, match="model identity does not match"
    ):
        app._load_engine_from_mlflow(reference)

    assert downloads == []


def test_cancelled_scoring_keeps_capacity_until_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_operation():
        try:
            started.set()
            assert release.wait(timeout=5)
            return "done"
        finally:
            finished.set()

    monkeypatch.setattr(
        app,
        "SCORING_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )

    async def scenario():
        task = asyncio.create_task(
            app._run_bounded_operation(
                app.SCORING_OPERATION_LIMITER,
                slow_operation,
                (),
                busy_error_code="SCORING_BUSY",
                busy_message="busy",
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0.05)
        assert task.done()
        assert task.cancelled()
        with pytest.raises(app.APIError) as error_info:
            await app._run_bounded_operation(
                app.SCORING_OPERATION_LIMITER,
                lambda: "second",
                (),
                busy_error_code="SCORING_BUSY",
                busy_message="busy",
            )
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.sleep(0)
        result = await app._run_bounded_operation(
            app.SCORING_OPERATION_LIMITER,
            lambda: "second",
            (),
            busy_error_code="SCORING_BUSY",
            busy_message="busy",
        )
        return error_info.value, result

    error, result = asyncio.run(scenario())

    assert error.status_code == 503
    assert error.error_code == "SCORING_BUSY"
    assert result == "second"
