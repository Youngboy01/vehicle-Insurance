import asyncio
import json
import math
import socket
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
    asyncio.run(
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


def test_predict_uses_bounded_complete_scoring_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine("model")
    monkeypatch.setattr(app, "current_engine", engine)
    monkeypatch.setattr(
        app,
        "SCORING_OPERATION_LIMITER",
        anyio.CapacityLimiter(1),
    )
    monkeypatch.setattr(
        app,
        "_score_raw_dataframe",
        lambda _engine, _dataframe: [0.81234],
    )

    response = asyncio.run(app.predict(_record()))

    assert response.response_propensity_score == 0.8123
    assert response.model_dump()["probability"] == 0.8123
    assert response.model_version == engine.model_version


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
                    for customer_id in range(1, 10_001)
                ],
                "top_k_count": 1,
            }
        ).encode("utf-8")
        rank_task = asyncio.create_task(
            _asgi_request("/rank", method="POST", body=body)
        )
        assert await asyncio.to_thread(started.wait, 10)

        started_at = time.perf_counter()
        ready_status, ready_body = await _asgi_request("/health/ready")
        readiness_elapsed = time.perf_counter() - started_at

        monkeypatch.setattr(app, "current_engine", replacement_engine)
        release.set()
        rank_status, rank_body = await rank_task
        return (
            ready_status,
            json.loads(ready_body),
            readiness_elapsed,
            rank_status,
            json.loads(rank_body),
        )

    ready_status, ready, readiness_elapsed, rank_status, response = asyncio.run(
        scenario()
    )

    assert ready_status == 200
    assert ready["status"] == "ready"
    assert readiness_elapsed < 0.5
    assert rank_status == 200
    assert response["model_id"] == old_engine.model_id
    print(
        "A06 readiness during scoring: "
        f"batch_size=10000 concurrency=1 native_threads={app.SERVING_MODEL_THREADS} "
        f"latency_ms={readiness_elapsed * 1000:.3f}"
    )


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

        started_at = time.perf_counter()
        ready_status, ready_body = await _asgi_request("/health/ready")
        readiness_elapsed = time.perf_counter() - started_at
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
        return reload_status, json.loads(reload_body), readiness_elapsed

    reload_status, response, readiness_elapsed = asyncio.run(successful_reload())

    assert app.current_engine is replacement_engine
    assert reload_status == 200
    assert response["model_id"] == replacement_engine.model_id
    assert readiness_elapsed < 0.5
    print(
        "A06 readiness during reload: "
        f"batch_size=not_applicable scoring_concurrency=0 "
        f"native_threads={app.SERVING_MODEL_THREADS} "
        f"latency_ms={readiness_elapsed * 1000:.3f}"
    )

    def failed_load():
        raise ConnectionError("synthetic MLflow outage")

    monkeypatch.setattr(app, "_load_engine_from_mlflow", failed_load)

    async def failed_reload():
        with pytest.raises(app.APIError) as error_info:
            await app.reload_model()
        return error_info.value

    error = asyncio.run(failed_reload())

    assert error.status_code == 503
    assert error.error_code == "DEPENDENCY_UNAVAILABLE"
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


def _mlflow_candidate_engine() -> tuple[app.PredictionEngine, float]:
    """Load the exact immutable candidate through the production MLflow loader."""

    started_at = time.perf_counter()
    engine = app._load_engine_from_mlflow(app.CURRENT_MODEL_REFERENCE)
    load_latency_ms = (time.perf_counter() - started_at) * 1000
    reference = app.CURRENT_MODEL_REFERENCE
    assert engine.model_id == reference.model_id
    assert engine.model_uri == reference.model_uri
    assert engine.source_run_id == reference.run_id
    return engine, load_latency_ms


def _http_request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    timeout: float = 5.0,
) -> dict[str, object]:
    request = Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    started_at = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read()
            status = response.status
        outcome = "success" if status == 200 else "failed"
        return {
            "outcome": outcome,
            "status": status,
            "body": response_body,
            "latency_ms": (time.perf_counter() - started_at) * 1000,
        }
    except HTTPError as error:
        return {
            "outcome": "failed",
            "status": error.code,
            "body": error.read(),
            "latency_ms": (time.perf_counter() - started_at) * 1000,
        }
    except (TimeoutError, socket.timeout):
        return {
            "outcome": "timeout",
            "status": None,
            "body": b"",
            "latency_ms": (time.perf_counter() - started_at) * 1000,
        }
    except URLError as error:
        outcome = (
            "timeout"
            if isinstance(error.reason, (TimeoutError, socket.timeout))
            else "connection_failure"
        )
        return {
            "outcome": outcome,
            "status": None,
            "body": b"",
            "latency_ms": (time.perf_counter() - started_at) * 1000,
        }
    except OSError:
        return {
            "outcome": "connection_failure",
            "status": None,
            "body": b"",
            "latency_ms": (time.perf_counter() - started_at) * 1000,
        }


def _start_local_http_server() -> tuple[app.uvicorn.Server, threading.Thread, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        port = probe_socket.getsockname()[1]

    config = app.uvicorn.Config(
        app.app,
        host="127.0.0.1",
        port=port,
        workers=1,
        lifespan="off",
        log_level="critical",
        access_log=False,
    )
    server = app.uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.perf_counter() + 10
    while time.perf_counter() < deadline:
        if not server_thread.is_alive():
            raise AssertionError("The local Uvicorn worker exited during startup")
        response = _http_request(base_url, "/health/ready", timeout=1)
        if response["status"] == 200:
            return server, server_thread, base_url
        time.sleep(0.01)
    server.should_exit = True
    server_thread.join(timeout=5)
    raise AssertionError("The local Uvicorn worker did not become ready")


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def test_a06_real_http_batch_readiness_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measure readiness during the real 10,000-row HTTP scoring path."""

    engine, mlflow_load_latency_ms = _mlflow_candidate_engine()
    monkeypatch.setattr(app, "current_engine", engine)
    monkeypatch.setattr(app, "SCORING_OPERATION_LIMITER", anyio.CapacityLimiter(1))

    scoring_started = threading.Event()
    scoring_finished = threading.Event()
    scoring_times: dict[str, float] = {}
    original_rank_builder = app._build_rank_response

    def observed_rank_builder(engine, request):
        scoring_times["started"] = time.perf_counter()
        scoring_started.set()
        try:
            return original_rank_builder(engine, request)
        finally:
            scoring_times["finished"] = time.perf_counter()
            scoring_finished.set()

    monkeypatch.setattr(app, "_build_rank_response", observed_rank_builder)
    request_body = json.dumps(
        {
            "records": [
                _record(customer_id).model_dump()
                for customer_id in range(1, 10_001)
            ],
            "top_k_count": 10_000,
        },
        separators=(",", ":"),
    ).encode("utf-8")

    server = None
    server_thread = None
    try:
        server, server_thread, base_url = _start_local_http_server()
        readiness_results: list[dict[str, object]] = []

        def probe_readiness_until_scoring_finishes() -> None:
            assert scoring_started.wait(timeout=15)
            with ThreadPoolExecutor(max_workers=16) as probe_pool:
                active = set()
                while not scoring_finished.is_set() or active:
                    while not scoring_finished.is_set() and len(active) < 16:
                        active.add(
                            probe_pool.submit(
                                _http_request,
                                base_url,
                                "/health/ready",
                                timeout=2,
                            )
                        )
                    if not active:
                        break
                    completed, active = wait(
                        active,
                        timeout=0.25,
                        return_when=FIRST_COMPLETED,
                    )
                    readiness_results.extend(task.result() for task in completed)

        probe_thread = threading.Thread(
            target=probe_readiness_until_scoring_finishes,
            daemon=True,
        )
        probe_thread.start()
        with ThreadPoolExecutor(max_workers=1) as client_pool:
            rank_future = client_pool.submit(
                _http_request,
                base_url,
                "/rank",
                method="POST",
                body=request_body,
                timeout=90,
            )
            assert scoring_started.wait(timeout=15)
            rank_result = rank_future.result(timeout=90)
        assert scoring_finished.wait(timeout=5)
        probe_thread.join(timeout=10)
        assert not probe_thread.is_alive()

        assert rank_result["status"] == 200
        rank_payload = json.loads(rank_result["body"])
        assert rank_payload["total_scored"] == 10_000
        assert rank_payload["selected_count"] == 10_000
        assert len(rank_payload["selected_customers"]) == 10_000

        assert len(readiness_results) >= 16
        successful = [
            result
            for result in readiness_results
            if result["outcome"] == "success"
        ]
        failed = [
            result
            for result in readiness_results
            if result["outcome"] == "failed"
        ]
        timeouts = [
            result
            for result in readiness_results
            if result["outcome"] == "timeout"
        ]
        connection_failures = [
            result
            for result in readiness_results
            if result["outcome"] == "connection_failure"
        ]
        latencies = [
            float(result["latency_ms"])
            for result in readiness_results
        ]
        report = {
            "mlflow_load_latency_ms": mlflow_load_latency_ms,
            "model_id": engine.model_id,
            "model_uri": engine.model_uri,
            "source_run_id": engine.source_run_id,
            "batch_size": 10_000,
            "selected_audience": 10_000,
            "uvicorn_workers": 1,
            "scoring_concurrency": 1,
            "native_model_threads": int(
                engine.model.named_steps["model"].n_jobs
            ),
            "target_latency_ms": 500,
            "request_count": len(readiness_results),
            "successful_probes": len(successful),
            "failed_probes": len(failed),
            "p50_latency_ms": _nearest_rank_percentile(latencies, 50),
            "p95_latency_ms": _nearest_rank_percentile(latencies, 95),
            "p99_latency_ms": _nearest_rank_percentile(latencies, 99),
            "maximum_latency_ms": max(latencies),
            "timeouts": len(timeouts),
            "connection_failures": len(connection_failures),
            "rank_http_latency_ms": float(rank_result["latency_ms"]),
            "scoring_worker_latency_ms": (
                scoring_times["finished"] - scoring_times["started"]
            )
            * 1000,
            "response_bytes": len(rank_result["body"]),
        }
        assert report["successful_probes"] == report["request_count"]
        assert report["failed_probes"] == 0
        assert report["timeouts"] == 0
        assert report["connection_failures"] == 0
        assert report["p95_latency_ms"] < report["target_latency_ms"]
        print("A06.6 real HTTP readiness distribution: " + json.dumps(report, sort_keys=True))
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
            assert not server_thread.is_alive()
