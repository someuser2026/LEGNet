#!/usr/bin/env bash
set -euo pipefail

# Submit one PBS job per config in the planet Nearmap config directory.
#
# Examples:
#   tools/submit_planet_nearmap_jobs.sh
#   tools/submit_planet_nearmap_jobs.sh --email you@example.com
#   tools/submit_planet_nearmap_jobs.sh --config-dir configs/planet_nearmap_448
#   tools/submit_planet_nearmap_jobs.sh --inference-only
#   tools/submit_planet_nearmap_jobs.sh --wandb-project legnet-obb --dry-run
#
# Notes:
# - Uses tools/train_single_gpu_wandb.pbs.
# - INFERENCE_ONLY is passed through as env var to the PBS script.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIRS=(
    "configs/planet_nearmap_448"
)
PBS_SCRIPT="tools/train_single_gpu_wandb.pbs"

EMAIL_TO="z5428587@ad.unsw.edu.au"
WANDB_PROJECT=""
WORK_DIR_BASE=""
INFERENCE_ONLY="0"
INFERENCE_WALLTIME="02:00:00"
TEST_AFTER_TRAIN=""
TEST_CHECKPOINT=""
QSUB_RESOURCES="select=1:ncpus=6:ngpus=1:mem=32gb:gpu_model=A100"
DRY_RUN="0"
CUSTOM_CONFIG_DIRS="0"

usage() {
    cat <<'EOF'
Usage: tools/submit_planet_nearmap_jobs.sh [options]

Options:
  --config-dir <path>          Config directory to scan for .py files.
                               Repeat to include multiple directories.
                               If provided, overrides default directories.
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
        --config-dir)
            if [[ "$CUSTOM_CONFIG_DIRS" != "1" ]]; then
                CONFIG_DIRS=()
                CUSTOM_CONFIG_DIRS="1"
            fi
            CONFIG_DIRS+=("$2")
            shift 2
            ;;
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

if [[ ! -f "$PBS_SCRIPT" ]]; then
    echo "ERROR: PBS script not found: $PBS_SCRIPT" >&2
    exit 1
fi

CONFIGS=()
for config_dir in "${CONFIG_DIRS[@]}"; do
    if [[ ! -d "$config_dir" ]]; then
        echo "ERROR: Config directory not found: $config_dir" >&2
        exit 1
    fi
    while IFS= read -r cfg; do
        CONFIGS+=("$cfg")
    done < <(find "$config_dir" -maxdepth 1 -type f -name '*.py' | sort)
done

if [[ ${#CONFIGS[@]} -gt 0 ]]; then
    sorted_configs=()
    while IFS= read -r cfg; do
        sorted_configs+=("$cfg")
    done < <(printf '%s\n' "${CONFIGS[@]}" | sort -u)
    CONFIGS=("${sorted_configs[@]}")
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: No config files found in configured directories." >&2
    exit 1
fi

echo "Submitting ${#CONFIGS[@]} config(s) from:"
for config_dir in "${CONFIG_DIRS[@]}"; do
    echo "  - $config_dir"
done
echo "PBS script: $PBS_SCRIPT"
echo "Inference-only: $INFERENCE_ONLY"
if [[ "$INFERENCE_ONLY" = "1" ]]; then
    echo "Inference-only walltime override: $INFERENCE_WALLTIME"
fi
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

    cmd=(qsub -l "$QSUB_RESOURCES")
    if [[ "$INFERENCE_ONLY" = "1" ]]; then
        cmd+=(-l "walltime=${INFERENCE_WALLTIME}")
    fi
    cmd+=(-V -v "$env_vars" "$PBS_SCRIPT")
    echo "Config: $(basename "$cfg")"
    echo "  ${cmd[*]}"

    if [[ "$DRY_RUN" != "1" ]]; then
        "${cmd[@]}"
    fi
    echo
done
