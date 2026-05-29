# Pushing Worker Images to AWS ECR

`scripts/push_workers_to_ecr.sh` is the single script for building NimbusImage
worker Docker images and pushing them to a private AWS ECR registry. Given
valid AWS credentials, it:

1. Discovers workers from the `workers/` tree.
2. Cross-builds each for `linux/amd64` with `docker buildx`.
3. Builds and pushes any base image the worker needs (see
   [Base images](#base-images)), then redirects the worker's `FROM` to the
   ECR copy.
4. Pushes each worker under two tags: the short git SHA and `latest`.

It is a **local, laptop-driven workflow** — run it after sourcing production
AWS credentials. There is no CI job that pushes images.

---

## TL;DR

```bash
# 0. (macOS, once) install a modern bash
brew install bash

# 1. Auth — sets AWS keys + session token (region defaults to us-east-1)
source ../AWSDeploy/aws_credentials_prod.sh

# 2. See the exact worker names
/opt/homebrew/bin/bash scripts/push_workers_to_ecr.sh --list

# 3. Build & push one or more workers
/opt/homebrew/bin/bash scripts/push_workers_to_ecr.sh blob_metrics_worker connect_timelapse

# Reuse a base image already in ECR (skip the slow base rebuild)
/opt/homebrew/bin/bash scripts/push_workers_to_ecr.sh --skip-base blob_metrics_worker

# Preview the exact commands without running them
/opt/homebrew/bin/bash scripts/push_workers_to_ecr.sh --dry-run --all
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **bash 4+** | macOS ships bash 3.2. `brew install bash`, then invoke `/opt/homebrew/bin/bash scripts/push_workers_to_ecr.sh ...`. The script refuses to run on bash 3. |
| **Docker + buildx** | Docker Desktop is fine. The script creates a `docker-container` buildx builder named `nimbus-ecr-builder` on first run. |
| **AWS CLI v2** | Used for `sts get-caller-identity`, ECR login, and repo creation. |
| **ECR permissions** | The credentials must allow `ecr:GetAuthorizationToken`, `ecr:CreateRepository`, `ecr:DescribeRepositories`, and push (`ecr:*` / PowerUser is simplest). |

## Authentication

Credentials come from the AWSDeploy repo's sourced env script:

```bash
source ../AWSDeploy/aws_credentials_prod.sh
```

This exports `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
`AWS_SESSION_TOKEN` (SSO short-lived creds). It does **not** set a region, so
the script defaults to `us-east-1` (override with `--region` or `$AWS_REGION`).
The account ID is read automatically from `aws sts get-caller-identity`.

> **Gotcha — shells don't share env.** If you run the script from a different
> shell/terminal than the one you sourced creds in, it won't see them. Source
> the creds in the same shell, or re-source before each run.

The current prod registry is account **`677276105390`**, region **`us-east-1`**.

## Options

```
--all              Build and push every discovered worker (asks first)
--list             Print every discovered worker name and exit
--skip-base        Don't (re)build base images; assume they're already in ECR
--region REGION    AWS region (default: $AWS_REGION or us-east-1)
--prefix PREFIX    ECR repo prefix (default: nimbus)
--platform PLAT    Build platform (default: linux/amd64)
--no-cache         Disable the local buildx cache for this run
--dry-run          Print actions but run nothing
-y, --yes          Skip confirmation prompts (creating ECR repos, --all)
-h, --help         Show usage
```

## Image naming / tags

```
<acct>.dkr.ecr.<region>.amazonaws.com/<prefix>/<category>/<worker>:<short-git-sha>
<acct>.dkr.ecr.<region>.amazonaws.com/<prefix>/<category>/<worker>:latest
```

- `<category>` is `annotations` or `properties`.
- `<prefix>` defaults to `nimbus`; base images go under `<prefix>/base/<base>`.
- The SHA tag gets a `-dirty` suffix when the git working tree has uncommitted
  changes, so you always know an image was built from clean source.
- ECR repos are created on first push (confirmation prompt; `-y` to skip).
  Repos are created with tag mutability `MUTABLE` and `scanOnPush=true`.

## Base images

Workers build `FROM nimbusimage/<base>:latest` (`worker-base`,
`image-processing-base`, or `test-worker-base`). Those images are **not on any
public registry**, and any copy on your machine is built for the **host
architecture** (arm64 on Apple Silicon), so they can't satisfy an amd64
cross-build.

The script handles this automatically:

1. Builds the needed base for the target platform from
   `workers/base_docker_images/Dockerfile.<base>`.
2. Pushes it to `<prefix>/base/<base>` in ECR.
3. Redirects the worker's `FROM` to that ECR image with
   `docker buildx --build-context nimbusimage/<base>:latest=docker-image://<ecr>/<prefix>/base/<base>:latest`
   — **no worker Dockerfile edits required**.

A base is built at most once per run. Use `--skip-base` on later runs to reuse
a base already pushed to ECR (the base build is the slowest step on macOS).

## Build context auto-detection

Worker Dockerfiles in this repo are inconsistent about build context:

- Most COPY repo-root-relative paths (e.g.
  `COPY ./workers/<cat>/<name>/entrypoint.py /`) and need the **repo root** as
  context.
- A few (cellpose, stardist, random_point, …) COPY worker-dir-relative paths
  (e.g. `COPY ./environment.yml /`) and need the **worker directory** as
  context.

The script auto-detects the correct context per worker by checking where each
`COPY`/`ADD` source actually resolves on disk, so both layouts build correctly
under `--all`. (This fixes the original limitation where everything was built
from the repo root and worker-dir Dockerfiles failed.)

## Worker categories — what builds where

| Group | Base | Builds with this script? | Notes |
|---|---|---|---|
| Property/connection/connect workers | `nimbusimage/worker-base` | ✅ fast | Reuse the worker-base image. |
| Image-processing workers (crop, histogram_matching, registration, …) | `nimbusimage/image-processing-base` | ✅ | Needs the image-processing base built once. |
| Older "BASE_IMAGE" workers (random_point, line_length_worker, point_to_nearest_*_distance) | `ghcr.io/arjunrajlaboratory/base_x86_image` (public, default ARG) | ✅ | Pull a public x86 base, then `conda env update`. |
| Self-contained `ubuntu:jammy` workers | none (apt + conda from scratch) | ✅ slow | Each builds everything from scratch; slow under emulation. |
| **GPU / ML workers** (cellpose family, sam*, deepcell, stardist, condensatenet, deconwolf, cellori) | `nvidia/cuda:*` | ⚠️ **prefer native x86** | See below. |

### What we deliberately do **not** push

- **Test / sample workers** — `random_squares`, `sample_interface`,
  `test_*`. These are dev/testing artifacts, not deployed.
- **`_M1` variants** — `*_M1` Dockerfiles are CPU/Mac-development variants;
  the production `Dockerfile` is what gets deployed.

## Gotchas & specifics

- **Worker names are directory names**, not docker-compose service names.
  e.g. `blob_metrics_worker` (directory), not `blob_metrics` (compose service).
  Always confirm with `--list`.
- **macOS = QEMU emulation.** Building `linux/amd64` on Apple Silicon runs the
  build under emulation. File-copy-only workers are fast; anything with a
  conda solve / `apt install` is **much slower** (minutes to tens of minutes
  each).
- **GPU/ML workers should be built on a real x86_64 machine** (an EC2 amd64
  instance, an amd64 CI runner, or a remote buildx builder) using
  `build_machine_learning_workers.sh`. Emulating CUDA/torch builds on a Mac is
  impractically slow and can fail. The script can *discover* them and detects
  their context correctly, but native amd64 is strongly preferred for these.
- **Piscis** has a non-standard `predict/` + `train/` layout and is not
  discovered by this script; build it via `build_machine_learning_workers.sh`.
- **Per-worker buildx cache** lives under `.cache/buildx/<worker>/` (and
  `.cache/buildx/base-<base>/`) so reruns are fast; `--no-cache` disables it.

## Verifying in the AWS console

ECR → **Private registry → Repositories** (region `us-east-1`, account
`677276105390`). Each `nimbus/...` repo lists its image tags, pushed time,
size, and scan-on-push findings. Open an image digest to confirm the
architecture is `linux/amd64`.

Direct link (must be signed into the same account via SSO):
`https://us-east-1.console.aws.amazon.com/ecr/private-registry/repositories?region=us-east-1`

Or from the CLI:

```bash
aws ecr describe-repositories --region us-east-1 \
  --query "repositories[?starts_with(repositoryName,'nimbus/')].repositoryName" --output text

# confirm architecture of a pushed image
docker buildx imagetools inspect \
  677276105390.dkr.ecr.us-east-1.amazonaws.com/nimbus/properties/blob_metrics_worker:latest
```
