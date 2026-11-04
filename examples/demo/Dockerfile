# The demo image. Build from the repository root: docker build -f examples/demo/Dockerfile -t memory-kernel-demo .
FROM python:3.12-slim

# build-essential: a few transitive wheels may compile. CPU torch: the default wheel carries gigabytes of CUDA.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<3"
WORKDIR /repo
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir ".[build,parser,serve,ops]"
ENV MK_DATA_DIR=/data
RUN mk setup --data-dir /data && mk setup --check --data-dir /data
COPY examples ./examples
USER nobody
