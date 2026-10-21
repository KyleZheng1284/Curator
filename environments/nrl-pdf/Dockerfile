# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VER=13.0.0
ARG LINUX_VER=ubuntu24.04

FROM nvidia/cuda:${CUDA_VER}-cudnn-devel-${LINUX_VER} AS cuda


########################################################################
# Base image
########################################################################

FROM cuda AS build

ARG CURATOR_ENV=ci
ENV CURATOR_ENVIRONMENT=${CURATOR_ENV}

ENV NVIDIA_PRODUCT_NAME="NeMo Curator"

# Install base dependency
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV DEBIAN_FRONTEND=noninteractive
# gnupg upgrade Address CVE-2025-68973
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    libmagic1t64 \
    vim && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python-is-python3 && \
    apt-get install -y --only-upgrade gnupg libunbound8 openssl libssl3t64 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Replace any bundled Nsight Systems with the pinned, patched devtools release.
# Keep the repository key scoped to this source.
RUN rm -rf /usr/local/cuda*/NsightSystems-cli-* /opt/nvidia/nsight-systems* && \
    installed_nsys=$(dpkg-query -W -f='${binary:Package} ${db:Status-Status}\n' \
        | awk '$1 ~ /^nsight-systems/ && $2 == "installed" {print $1}') && \
    if [ -n "${installed_nsys}" ]; then \
        apt-get remove --purge -y --allow-change-held-packages ${installed_nsys}; \
    fi && \
    NSYS_VERSION="2026.5.1.161-265138896106v0" && \
    NSYS_RELEASE=$(echo "${NSYS_VERSION}" | cut -d. -f1-3) && \
    NSYS_REPO="https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/lsb-release; echo "${DISTRIB_RELEASE}" | tr -d .)/$(dpkg --print-architecture)" && \
    curl -fsSL "${NSYS_REPO}/nvidia.pub" -o /usr/share/keyrings/nvidia-devtools.asc && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.asc] ${NSYS_REPO}/ /" \
        > /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends "nsight-systems-cli-${NSYS_RELEASE}=${NSYS_VERSION}" && \
    nsys --version && \
    test "$(dpkg-query -W -f='${Version}' "nsight-systems-cli-${NSYS_RELEASE}")" = "${NSYS_VERSION}" && \
    rm -f /etc/apt/sources.list.d/nvidia-devtools.list /usr/share/keyrings/nvidia-devtools.asc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Install uv
ENV UV_VERSION="0.12.1"
RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_CACHE_DIR=/opt/uv_cache
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
ENV UV_LINK_MODE=copy
RUN uv venv ${UV_PROJECT_ENVIRONMENT} --python /usr/bin/python3.12 --system-site-packages --seed


FROM build AS curator_nrl_pdf
COPY wheels /opt/profile/wheels
COPY pyproject.toml uv.lock /opt/profile/
WORKDIR /opt/profile
RUN uv sync --locked --no-default-groups --no-progress && uv pip check
RUN python -c "from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage; stage = InterleavedAspectRatioFilterStage(preserve_metadata_only_samples=True); assert stage.preserve_metadata_only_samples"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
LABEL org.opencontainers.image.title="Experimental Curator NRL PDF CUDA 13 profile"
CMD ["python", "-c", "import nemo_retriever,nemo_curator; print('NRL and Curator imports passed')"]
