#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

for command in docker nvidia-smi sha256sum; do
  command -v "${command}" >/dev/null || fail "missing ${command}"
done

docker compose version >/dev/null || fail "Docker Compose plugin is unavailable"
docker info >/dev/null || fail "Docker daemon is unavailable"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [ "${gpu_count}" -lt 4 ]; then
  fail "four GPUs are required; found ${gpu_count}"
fi

model_path="${MODEL_PATH:-/srv/ai/models/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid}"
model_files=(
  "config.json:254974797e9f455716a30ab5505ba68272181b20b58a3693e54f94fb8056f3ef"
  "model.safetensors.index.json:6eb773222d932418dd0530c63aca498f86ef424da2a4526ccba76b59726da234"
  "tokenizer_config.json:98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"
  "mxfp8_tier_nokvb.json:ebcd6087180033d4512fafa5f154f4fecfbc1ee5e5051448f34859cccc4430f0"
)
for entry in "${model_files[@]}"; do
  file="${entry%%:*}"
  expected="${entry#*:}"
  path="${model_path}/${file}"
  test -f "${path}" || fail "missing ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  test "${actual}" = "${expected}" || fail "checksum mismatch for ${path}"
done

printf '%s\n' 'GPU topology:'
nvidia-smi topo -m
printf '%s\n' 'Peer-read capability:'
peer_matrix="$(nvidia-smi topo -p2p r)"
printf '%s\n' "${peer_matrix}"

IFS=',' read -r -a selected_gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [ "${#selected_gpus[@]}" -ne 4 ]; then
  fail "this profile requires exactly four CUDA_VISIBLE_DEVICES entries"
fi
for source in "${selected_gpus[@]}"; do
  [[ "${source}" =~ ^[0-9]+$ ]] || fail "preflight expects numeric GPU indices"
  for destination in "${selected_gpus[@]}"; do
    [ "${source}" = "${destination}" ] && continue
    status="$(printf '%s\n' "${peer_matrix}" | awk \
      -v row="GPU${source}" -v column="$((destination + 2))" \
      '$1 == row {print $column}')"
    [ "${status}" = "OK" ] || \
      fail "peer reads are ${status:-unknown} from GPU${source} to GPU${destination}"
  done
done

if find /sys/kernel/iommu_groups -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  warn "IOMMU groups are active; the validated host used amd_iommu=off iommu=off"
fi

if command -v lspci >/dev/null; then
  if lspci -vv 2>/dev/null | grep -Eq 'ACSCtl:.*(SrcValid\+|TransBlk\+|ReqRedir\+|CmpltRedir\+)'; then
    warn "ACS redirect appears enabled; verify that peer traffic is not redirected"
  fi
fi

available_gib="$(df -BG --output=avail "${model_path}" | tail -1 | tr -dc '0-9')"
if [ -n "${available_gib}" ] && [ "${available_gib}" -lt 60 ]; then
  warn "less than 60 GiB is free on the model filesystem"
fi

printf '%s\n' 'Power and link state:'
nvidia-smi --query-gpu=index,name,power.limit,pci.bus_id,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader
printf '%s\n' 'Preflight complete. Review any warnings before launch.'
