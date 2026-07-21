ARG BASE_IMAGE=voipmonitor/vllm@sha256:41078d5fac36b802e0f3798057a4b383c24296cb8570f6f4fc138c34d8cfa303

FROM ${BASE_IMAGE} AS sources

ARG VLLM_REPOSITORY=https://github.com/FujitsuPolycom/vllm.git
ARG VLLM_BASE_REF=7ea567a2458a4800a6a0e3e0a6ba41fcbd00d146
ARG VLLM_REF=a81a0811314e23bc6b2baa69f34247aacea71e23
ARG SPARKINFER_REPOSITORY=https://github.com/FujitsuPolycom/sparkinfer.git
ARG SPARKINFER_REF=a82fc76f9a2321e3cfd3065a9fa78452a27ddf9f

USER root

RUN set -eux; \
    git clone --filter=blob:none "${VLLM_REPOSITORY}" /src/vllm; \
    git -C /src/vllm checkout --detach "${VLLM_REF}"; \
    test "$(git -C /src/vllm rev-parse HEAD)" = "${VLLM_REF}"; \
    git -C /src/vllm cat-file -e "${VLLM_BASE_REF}^{commit}"; \
    mkdir -p /src/vllm-overlay; \
    git -C /src/vllm diff --name-only --diff-filter=ACMRTUXB \
      "${VLLM_BASE_REF}..${VLLM_REF}" -- vllm \
      | while IFS= read -r path; do \
          mkdir -p "/src/vllm-overlay/$(dirname "${path}")"; \
          cp "/src/vllm/${path}" "/src/vllm-overlay/${path}"; \
        done; \
    test "$(find /src/vllm-overlay/vllm -type f | wc -l)" = "39"; \
    git clone --filter=blob:none "${SPARKINFER_REPOSITORY}" /src/sparkinfer; \
    git -C /src/sparkinfer checkout --detach "${SPARKINFER_REF}"; \
    test "$(git -C /src/sparkinfer rev-parse HEAD)" = "${SPARKINFER_REF}"; \
    python -m pip wheel --no-cache-dir --no-deps \
      --wheel-dir /dist /src/sparkinfer

FROM ${BASE_IMAGE}

ARG VLLM_REF=a81a0811314e23bc6b2baa69f34247aacea71e23
ARG SPARKINFER_REF=a82fc76f9a2321e3cfd3065a9fa78452a27ddf9f
ARG RUN_CPU_TESTS=1

USER root

COPY --from=sources /dist/ /opt/release/dist/
COPY --from=sources /src/vllm-overlay/vllm/ /opt/release/vllm/
COPY --from=sources /src/vllm/tests/ /opt/release-tests/vllm-tests/
COPY --from=sources /src/sparkinfer/tests/ /opt/release-tests/sparkinfer-tests/

RUN set -eux; \
    python -m pip install --no-cache-dir --upgrade \
      nvidia-cutlass-dsl==4.6.0 \
      nvidia-cutlass-dsl-libs-base==4.6.0 \
      nvidia-cutlass-dsl-libs-core==4.6.0 \
      nvidia-cutlass-dsl-libs-cu12==4.6.0 \
      nvidia-cutlass-dsl-libs-cu13==4.6.0 \
      quack-kernels==0.6.1 \
      pytest==8.4.2; \
    python -m pip install --no-cache-dir --no-deps --force-reinstall \
      /opt/release/dist/sparkinfer-*.whl; \
    for root in \
      /opt/venv/lib/python3.12/site-packages/vllm \
      /opt/vllm/vllm; do \
        test -d "${root}"; \
        cp -a /opt/release/vllm/. "${root}/"; \
    done; \
    python -c 'from sparkinfer.comm.pcie import SelectedRecordCopyExchange, SelectedRecordExchange, SelectedRecordExchangeInitializationError; assert SelectedRecordCopyExchange and SelectedRecordExchange and SelectedRecordExchangeInitializationError'; \
    test -x /usr/local/bin/serve-gilded-gnosis.sh; \
    test -x /usr/local/bin/serve-glm52-hybrid-v19.sh; \
    python -m compileall -q \
      /opt/venv/lib/python3.12/site-packages/sparkinfer \
      /opt/venv/lib/python3.12/site-packages/vllm; \
    mkdir -p /opt/sparkinfer/comm/pcie; \
    cp /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/*.cu \
      /opt/sparkinfer/comm/pcie/; \
    cp /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_selected_records*.py \
      /opt/sparkinfer/comm/pcie/; \
    mkdir -p /opt/release-tests/sparkinfer/comm/pcie; \
    cp /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_selected_records*.cu \
      /opt/venv/lib/python3.12/site-packages/sparkinfer/comm/pcie/pcie_selected_records*.py \
      /opt/release-tests/sparkinfer/comm/pcie/; \
    if [ "${RUN_CPU_TESTS}" = "1" ]; then \
      python -m pytest -q -p no:cacheprovider \
        /opt/release-tests/sparkinfer-tests/comm/test_pcie_selected_records.py \
        /opt/release-tests/sparkinfer-tests/comm/test_pcie_selected_records_ce.py \
        /opt/release-tests/sparkinfer-tests/comm/test_pcie_selected_records_gpu.py \
        /opt/release-tests/sparkinfer-tests/comm/test_pcie_selected_records_ce_gpu.py \
        /opt/release-tests/sparkinfer-tests/comm/test_pcie_dcp_a2a.py; \
      python -m pytest -q -p no:cacheprovider \
        /opt/release-tests/vllm-tests/v1/attention/test_b12x_sparse_ckv_decode_policy.py \
        /opt/release-tests/vllm-tests/v1/attention/test_b12x_ckv_prefetch_policy.py \
        /opt/release-tests/vllm-tests/v1/attention/test_indexer_dcp_localize.py \
        /opt/release-tests/vllm-tests/v1/worker/test_workspace.py; \
      python -m pytest -q -p no:cacheprovider \
        /opt/release-tests/vllm-tests/v1/core/test_kv_cache_utils.py \
        -k 'lockstep_mla or mixed_dcp_replicated_groups'; \
    fi; \
    rm -rf /opt/release /opt/release-tests /opt/sparkinfer

COPY --chmod=0755 serve.sh /usr/local/bin/serve-glm52-sparse-ckv.sh

LABEL org.opencontainers.image.source="https://github.com/FujitsuPolycom/vllm" \
      org.opencontainers.image.revision="${VLLM_REF}" \
      ai01.hardware="ASUS WRX90E-SAGE SE; Threadripper PRO 9965WX; 128 GiB RAM; 4x RTX PRO 6000 Blackwell 96GB at 400W" \
      ai01.base="voipmonitor/vllm@sha256:41078d5fac36b802e0f3798057a4b383c24296cb8570f6f4fc138c34d8cfa303" \
      ai01.vllm="FujitsuPolycom/vllm@${VLLM_REF}" \
      ai01.sparkinfer="FujitsuPolycom/sparkinfer@${SPARKINFER_REF}" \
      ai01.features="replicated indexer; depth-3 CKV prefetch; selected-record sparse decode; bulk copy-engine transport" \
      local-inference.cutlass_dsl.version="4.6.0" \
      local-inference.vllm.commit="${VLLM_REF}" \
      local-inference.vllm.repo="https://github.com/FujitsuPolycom/vllm.git"
