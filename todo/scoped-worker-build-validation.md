# TODO-005: Make scoped worker builds fail when the Compose service is absent

**Status:** Open
**Priority:** Medium
**Owner:** unassigned
**Related:** coordinate-validation hardening for the 2026-08-06 `cellposesam` crash

## Problem

`build_workers.sh` documents a scoped build-and-test command using a worker name:

```bash
./build_workers.sh --build-and-run-tests cellposesam
```

On 2026-08-07 that command rebuilt all three base images, then printed:

```text
no such service: cellposesam
no such service: cellposesam_test
no such service: cellposesam_test
Process completed!
```

and exited with status 0. `cellposesam` is not registered in
`docker-compose.yml`, and the script neither validates the requested service nor
propagates the failing `docker compose` status. This makes an omitted test look
like a successful scoped verification after an expensive base-image build.

## Recommended resolution

1. Validate the requested service against
   `docker compose --profile "*" config --services` before building any base
   images.
2. Exit nonzero with an actionable message when the worker or its `_test`
   service is absent.
3. Decide whether ML workers such as `cellposesam` should be registered in
   Compose or explicitly routed to `build_machine_learning_workers.sh` and
   native worker tests.
4. Add shell-level regression coverage proving a missing scoped service cannot
   reach `Process completed!` or return status 0.

## Current verification path

Until this is fixed, run `workers/annotations/cellposesam/tests` natively for
its current unit coverage. The shared `worker_client/tests` suite covers the
coordinate preflight added for the production crash.
