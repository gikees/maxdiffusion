#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <version_number> <slice_count> [type|name|zone|series|flags...]"
    echo "  type   : reserved|spot (default: reserved) or -t/--type"
    echo "  name   : TPU name prefix (default: solaris) or -n/--name"
    echo "  zone   : e.g. us-east5-a (default: us-east5-a) or -z/--zone"
    echo "  series : e.g. v4, v5p, v5e, v6e (default: v5p) or -s/--series"
    echo "Flags can be mixed with positional tokens (any order):"
    echo "  --zone Z | --zone=Z | -z Z"
    echo "  --series S | --series=S | -s S"
    echo "  --type T | --type=T | -t T (reserved|spot)"
    echo "  --name N | --name=N | -n N"
    echo "Examples:"
    echo "  $0 2 256                         # reserved, name 'solaris', zone us-east5-a, series v5p"
    echo "  $0 4 256 spot myproj             # spot, name 'myproj', zone us-east5-a, series v5p"
    echo "  $0 3 512 myproj us-east5-b v4    # reserved, name 'myproj', zone us-east5-b, series v4"
    echo "  $0 128-1 128 --zone us-east5-b --series v6e"
    echo "This will create a <series>-<slice_count> TPU with name <name>-<series>-<version_number>_<type>"
    exit 1
fi

# --- Required ---
VERSION_NUMBER=$1
SLICE_COUNT=$2
shift 2

# --- Defaults ---
TPU_TYPE="reserved"
TPU_NAME_PREFIX="solaris"
ZONE="us-east5-a"
TPU_SERIES="v5p"

# --- Parse optional args (any order): remaining can be type|name|zone|series or flags ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    # Long/short flags with separate values
    -z|--zone)
      ZONE="$2"; shift 2;;
    --zone=*)
      ZONE="${1#*=}"; shift;;
    -s|--series)
      TPU_SERIES="$2"; shift 2;;
    --series=*)
      TPU_SERIES="${1#*=}"; shift;;
    -t|--type)
      TPU_TYPE="$2"; shift 2;;
    --type=*)
      TPU_TYPE="${1#*=}"; shift;;
    -n|--name)
      TPU_NAME_PREFIX="$2"; shift 2;;
    --name=*)
      TPU_NAME_PREFIX="${1#*=}"; shift;;
    # Unflagged tokens (keep legacy behavior)
    spot|reserved)
      TPU_TYPE="$1"; shift;;
    v4|v5|v5e|v5p|v5lite*|v6e)
      TPU_SERIES="$1"; shift;;
    # zone pattern like us-east5-a, europe-west4-b, asia-southeast1-c
    [a-z][a-z0-9-]*-[a-z0-9-]*-[a-z])
      ZONE="$1"; shift;;
    # name (first free-form token that isn't already set via --name)
    *)
      if [ "$TPU_NAME_PREFIX" = "solaris" ]; then
        TPU_NAME_PREFIX="$1";
      fi
      shift;;
  esac
done


# Create TPU name based on name prefix, version number and type
TPU_NAME="${TPU_NAME_PREFIX}-${TPU_SERIES}-${VERSION_NUMBER}-${TPU_TYPE}"

# Build base command
if [ "$TPU_SERIES" = "v5e" ]; then
    ACCELERATOR_TYPE="v5litepod-${SLICE_COUNT}"
else
    ACCELERATOR_TYPE="${TPU_SERIES}-${SLICE_COUNT}"
fi
if [ "$TPU_SERIES" = "v5p" ]; then
    RUNTIME_VERSION="v2-alpha-tpuv5"
elif [ "$TPU_SERIES" = "v5e" ]; then
    RUNTIME_VERSION="v2-alpha-tpuv5-lite"
elif [ "$TPU_SERIES" = "v6e" ]; then
    RUNTIME_VERSION="v2-alpha-tpuv6e"
else
    RUNTIME_VERSION="tpu-ubuntu2204-base"
fi

# Build command safely (no eval) and respect type
cmd=(
  gcloud compute tpus queued-resources create "$TPU_NAME"
  --node-id "$TPU_NAME"
  --project nyu-vision-lab
  --zone "$ZONE"
  --accelerator-type "$ACCELERATOR_TYPE"
  --runtime-version "$RUNTIME_VERSION"
)
# Always create as spot regardless of TPU_TYPE (name still reflects requested type)
cmd+=(--spot)

echo "Running command:"
printf '%q ' "${cmd[@]}"; echo
"${cmd[@]}"

# Print confirmation message
echo "TPU resource creation initiated:"
echo "Name: ${TPU_NAME}"
echo "Type: ${TPU_TYPE}"
echo "Accelerator Type: ${ACCELERATOR_TYPE}"
