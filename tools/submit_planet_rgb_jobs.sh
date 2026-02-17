#!/usr/bin/env bash
set -euo pipefail

# Submit one PBS job per config in configs/planet_rgb.
#
# Examples:
#   tools/submit_planet_rgb_jobs.sh
#   tools/submit_planet_rgb_jobs.sh --email you@example.com
#   tools/submit_planet_rgb_jobs.sh --inference-only
#   tools/submit_planet_rgb_jobs.sh --wandb-project legnet-obb --dry-run
#
# Notes:
# - Uses tools/train_single_gpu_wandb.pbs.
# - INFERENCE_ONLY is passed through as env var to the PBS script.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="configs/planet_rgb"
PBS_SCRIPT="tools/train_single_gpu_wandb.pbs"

EMAIL_TO="z5428587@ad.unsw.edu.au"
WANDB_PROJECT=""
WORK_DIR_BASE=""
INFERENCE_ONLY="0"
TEST_AFTER_TRAIN=""
TEST_CHECKPOINT=""
QSUB_RESOURCES="select=1:ncpus=6:ngpus=1:mem=32gb:gpu_model=A100"
DRY_RUN="0"

usage() {
    cat <<'EOF'
Usage: tools/submit_planet_rgb_jobs.sh [options]

Options:
  --email <addr>              Email for PBS failure notifications.
  --wandb-project <name>      Optional W&B project override.
  --work-dir-base <path>      Optional WORK_DIR_BASE override.
  --inference-only            Set INFERENCE_ONLY=1 (skip training, run test only).
  --test-after-train <0|1>    Optional TEST_AFTER_TRAIN override.
  --test-checkpoint <path>    Optional TEST_CHECKPOINT override.
  --resources "<pbs string>"  qsub -l resources string.
  --dry-run                   Print qsub commands without submitting.
  --help                      Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --email)
            EMAIL_TO="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --work-dir-base)
            WORK_DIR_BASE="$2"
            shift 2
            ;;
        --inference-only)
            INFERENCE_ONLY="1"
            shift
            ;;
        --test-after-train)
            TEST_AFTER_TRAIN="$2"
            shift 2
            ;;
        --test-checkpoint)
            TEST_CHECKPOINT="$2"
            shift 2
            ;;
        --resources)
            QSUB_RESOURCES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: Config directory not found: $CONFIG_DIR" >&2
    exit 1
fi

if [[ ! -f "$PBS_SCRIPT" ]]; then
    echo "ERROR: PBS script not found: $PBS_SCRIPT" >&2
    exit 1
fi

CONFIGS=()
while IFS= read -r cfg; do
    CONFIGS+=("$cfg")
done < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.py' | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: No config files found in $CONFIG_DIR" >&2
    exit 1
fi

echo "Submitting ${#CONFIGS[@]} config(s) from: $CONFIG_DIR"
echo "PBS script: $PBS_SCRIPT"
echo "Inference-only: $INFERENCE_ONLY"
echo "Dry-run: $DRY_RUN"
echo

for cfg in "${CONFIGS[@]}"; do
    rel_cfg="${cfg#${REPO_ROOT}/}"
    env_vars="CONFIG_YAML=${rel_cfg},EMAIL_TO=${EMAIL_TO},INFERENCE_ONLY=${INFERENCE_ONLY}"

    if [[ -n "$WANDB_PROJECT" ]]; then
        env_vars="${env_vars},WANDB_PROJECT=${WANDB_PROJECT}"
    fi
    if [[ -n "$WORK_DIR_BASE" ]]; then
        env_vars="${env_vars},WORK_DIR_BASE=${WORK_DIR_BASE}"
    fi
    if [[ -n "$TEST_AFTER_TRAIN" ]]; then
        env_vars="${env_vars},TEST_AFTER_TRAIN=${TEST_AFTER_TRAIN}"
    fi
    if [[ -n "$TEST_CHECKPOINT" ]]; then
        env_vars="${env_vars},TEST_CHECKPOINT=${TEST_CHECKPOINT}"
    fi

    cmd=(qsub -l "$QSUB_RESOURCES" -V -v "$env_vars" "$PBS_SCRIPT")
    echo "Config: $(basename "$cfg")"
    echo "  ${cmd[*]}"

    if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
    fi
    echo
done
