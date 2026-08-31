#!/usr/bin/env python3
"""Run a NimbusImage worker container with a short-lived Girder token."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
DATASET_VIEW_RE = re.compile(r"(?:^|/)datasetView/([^/?#]+)")


class RunError(RuntimeError):
    """An expected, safely reportable worker-run failure."""


def _api_url(api_root: str, path: str) -> str:
    return f"{api_root.rstrip('/')}/{path.lstrip('/')}"


def _request_json(
    api_root: str,
    method: str,
    path: str,
    *,
    token: str | None = None,
    form: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    headers = {"Accept": "application/json"}
    data = None
    if token:
        headers["Girder-Token"] = token
    if form is not None:
        data = urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = Request(
        _api_url(api_root, path), data=data, headers=headers, method=method
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RunError(
            f"{method} {path} returned HTTP {exc.code}: {detail}"
        ) from None
    except URLError as exc:
        raise RunError(f"Could not reach Girder at {api_root}: {exc.reason}") from None
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RunError(f"{method} {path} did not return JSON") from exc


def _read_api_key(env_file: Path) -> str:
    value = os.environ.get("NIMBUS_API_KEY", "").strip()
    if value:
        return value
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise RunError(
            f"NIMBUS_API_KEY is unset and environment file {env_file} is missing"
        ) from None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, candidate = line.partition("=")
        if separator and key.strip() == "NIMBUS_API_KEY":
            candidate = candidate.strip()
            if (
                len(candidate) >= 2
                and candidate[0] == candidate[-1]
                and candidate[0] in {"'", '"'}
            ):
                candidate = candidate[1:-1]
            if candidate:
                return candidate
    raise RunError(f"NIMBUS_API_KEY is not defined in {env_file}")


def _load_parameters(path: Path) -> tuple[dict[str, Any], str]:
    try:
        parameters = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RunError(f"Parameters file {path} does not exist") from None
    except json.JSONDecodeError as exc:
        raise RunError(f"Parameters file {path} is invalid JSON: {exc}") from exc
    if not isinstance(parameters, dict):
        raise RunError("Worker parameters must be one JSON object")
    return parameters, json.dumps(parameters, separators=(",", ":"), sort_keys=True)


def _dataset_view_id(value: str) -> str:
    parsed = urlsplit(value)
    candidates = (parsed.fragment, parsed.path, value)
    for candidate in candidates:
        match = DATASET_VIEW_RE.search(candidate)
        if match:
            view_id = match.group(1)
            if OBJECT_ID_RE.fullmatch(view_id):
                return view_id
            raise RunError(f"Dataset-view URL contains an invalid ID: {view_id!r}")
    raise RunError(
        "Could not find '/datasetView/<id>' in the supplied dataset-view URL"
    )


def _mint_session_token(api_root: str, api_key: str) -> str:
    response = _request_json(
        api_root, "POST", "api_key/token", form={"key": api_key}
    )
    try:
        token = response["authToken"]["token"]
    except (KeyError, TypeError):
        raise RunError("Girder did not return authToken.token") from None
    if not isinstance(token, str) or not token:
        raise RunError("Girder returned an empty session token")
    return token


def _revoke_session_token(api_root: str, token: str) -> None:
    _request_json(api_root, "DELETE", "token/session", token=token)


def _resolve_dataset_id(args: argparse.Namespace, token: str) -> str:
    if args.dataset_id:
        return args.dataset_id
    view_id = args.dataset_view_id
    if args.dataset_view_url:
        view_id = _dataset_view_id(args.dataset_view_url)
    response = _request_json(
        args.host_api_url,
        "GET",
        f"dataset_view/{quote(view_id, safe='')}",
        token=token,
    )
    dataset_id = response.get("datasetId") if isinstance(response, dict) else None
    if not isinstance(dataset_id, str) or not OBJECT_ID_RE.fullmatch(dataset_id):
        raise RunError(f"Dataset view {view_id} has no valid datasetId")
    return dataset_id


def _validate_dataset_folder(api_root: str, dataset_id: str, token: str) -> None:
    response = _request_json(
        api_root, "GET", f"folder/{quote(dataset_id, safe='')}", token=token
    )
    if not isinstance(response, dict) or str(response.get("_id")) != dataset_id:
        raise RunError(f"Girder did not resolve dataset folder {dataset_id}")


def _list_folder_items(api_root: str, dataset_id: str, token: str) -> list[dict]:
    items: list[dict] = []
    limit = 50
    offset = 0
    while True:
        query = urlencode(
            {"folderId": dataset_id, "limit": limit, "offset": offset}
        )
        page = _request_json(api_root, "GET", f"item?{query}", token=token)
        if not isinstance(page, list):
            raise RunError("Girder item listing did not return a list")
        items.extend(item for item in page if isinstance(item, dict))
        if len(page) < limit:
            return items
        offset += len(page)


def _preflight(image: str, network: str) -> None:
    checks = (
        (["docker", "image", "inspect", image], f"Docker image {image}"),
        (["docker", "network", "inspect", network], f"Docker network {network}"),
    )
    for command, label in checks:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RunError(f"Could not execute Docker: {exc}") from None
        if result.returncode:
            detail = result.stderr.strip()[:500]
            raise RunError(f"{label} is unavailable: {detail}")


def _container_name(image: str) -> str:
    base = image.rsplit("/", 1)[-1].split(":", 1)[0]
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", base).strip("-.") or "worker"
    return f"nimbus-run-{base}-{os.getpid()}"


def _docker_command(
    args: argparse.Namespace,
    dataset_id: str,
    token: str,
    parameters_json: str,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.container_name or _container_name(args.image),
        "--network",
        args.docker_network,
    ]
    if args.gpus:
        command.extend(["--gpus", args.gpus])
    command.extend(
        [
            args.image,
            "--datasetId",
            dataset_id,
            "--apiUrl",
            args.container_api_url,
            "--token",
            token,
            "--request",
            "compute",
            "--parameters",
            parameters_json,
        ]
    )
    return command


def _redacted_command(command: list[str]) -> list[str]:
    redacted = list(command)
    try:
        token_index = redacted.index("--token") + 1
    except ValueError:
        return redacted
    if token_index < len(redacted):
        redacted[token_index] = "<short-lived-girder-token>"
    return redacted


def _summary_item(item: dict) -> dict[str, Any]:
    return {
        "id": item.get("_id"),
        "name": item.get("name"),
        "size": item.get("size"),
        "large_image": bool(item.get("largeImage")),
    }


def execute(args: argparse.Namespace) -> int:
    _, parameters_json = _load_parameters(args.parameters_file)
    dry_dataset_id = args.dataset_id or "<resolved-dataset-folder-id>"
    if args.dry_run:
        command = _docker_command(
            args,
            dry_dataset_id,
            "<short-lived-girder-token>",
            parameters_json,
        )
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "api_calls_made": False,
                    "docker_calls_made": False,
                    "command": shlex.join(_redacted_command(command)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    api_key = _read_api_key(args.env_file)
    token = ""
    try:
        token = _mint_session_token(args.host_api_url, api_key)
        dataset_id = _resolve_dataset_id(args, token)
        _validate_dataset_folder(args.host_api_url, dataset_id, token)
        _preflight(args.image, args.docker_network)
        before_items = _list_folder_items(args.host_api_url, dataset_id, token)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "mode": "preflight-only",
                        "dataset_id": dataset_id,
                        "image": args.image,
                        "existing_folder_items": len(before_items),
                        "worker_started": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        before_ids = {str(item.get("_id")) for item in before_items}
        command = _docker_command(args, dataset_id, token, parameters_json)
        try:
            result = subprocess.run(command, check=False)
        except OSError as exc:
            raise RunError(f"Could not start the worker container: {exc}") from None
        after_items = _list_folder_items(args.host_api_url, dataset_id, token)
        new_items = [
            _summary_item(item)
            for item in after_items
            if str(item.get("_id")) not in before_ids
        ]
        print(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "image": args.image,
                    "container_exit_code": result.returncode,
                    "new_folder_items": new_items,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if result.returncode:
            raise RunError(
                f"Worker container exited with status {result.returncode}"
            )
        return 0
    finally:
        if token:
            try:
                _revoke_session_token(args.host_api_url, token)
            except RunError as exc:
                print(
                    f"Warning: could not revoke the temporary Girder token: {exc}",
                    file=sys.stderr,
                )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Built worker image tag")
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--dataset-id", help="Girder dataset folder ID")
    dataset.add_argument("--dataset-view-id", help="Nimbus dataset-view ID")
    dataset.add_argument("--dataset-view-url", help="Nimbus dataset-view URL")
    parser.add_argument(
        "--parameters-file", required=True, type=Path, help="Complete params JSON"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="File containing NIMBUS_API_KEY (default: .env)",
    )
    parser.add_argument(
        "--host-api-url",
        default="http://localhost:8080/api/v1",
        help="Girder API URL reachable from the host",
    )
    parser.add_argument(
        "--container-api-url",
        default="http://girder:8080/api/v1",
        help="Girder API URL reachable from the worker container",
    )
    parser.add_argument(
        "--docker-network",
        default="nimbusimage_default",
        help="Docker network shared with Girder",
    )
    parser.add_argument("--container-name", help="Optional deterministic name")
    parser.add_argument(
        "--gpus", help="Value for Docker --gpus, for example 'all'"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate JSON and print a redacted command without API/Docker calls",
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check auth, dataset, image, and network without starting the worker",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        for field in ("dataset_id", "dataset_view_id"):
            value = getattr(args, field)
            if value and not OBJECT_ID_RE.fullmatch(value):
                raise RunError(f"--{field.replace('_', '-')} must be a 24-hex ID")
        return execute(args)
    except RunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
