#!/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
  cat <<EOF # remove the space between << and EOF, this is due to web plugin issue
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] [-d <path/to/fastq/folder>]
This script runs a local version of the ARCHS4 pipeline.
Available options:
-h, --help      Print this help and exit
-v, --verbose   Print script debug info
-d, --dir       Specify the directory in which FASTQ directories
                are located. Each biological sample should be in
                a unique directory within this parent directory.
-o, --output    Specify the output directory. By default this is
                the output directory in the same folder as the script.
-z, --compress  gzip all fastq files after alignment. Defaults to
                true.
-i, --getindex  Get kallisto index from online Archs4 project. Only necessary once.
                Defaults to false.
EOF
  exit
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
}

setup_colors() {
  if [[ -t 2 ]] && [[ -z "${NO_COLOR-}" ]] && [[ "${TERM-}" != "dumb" ]]; then
    NOFORMAT='\033[0m' RED='\033[0;31m' GREEN='\033[0;32m' ORANGE='\033[0;33m' BLUE='\033[0;34m' PURPLE='\033[0;35m' CYAN='\033[0;36m' YELLOW='\033[1;33m'
  else
    NOFORMAT='' RED='' GREEN='' ORANGE='' BLUE='' PURPLE='' CYAN='' YELLOW=''
  fi
}

msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

parse_params() {
  # default values of variables set from params
  output="$PWD/output"
  compress=true
  getindex=false

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    --no-color) NO_COLOR=1 ;;
    -d | --dir)
      dir="${2-}"
      shift
      ;;
    -o | --output) # example named parameter
      output="${2-}"
      shift
      ;;
    -z | --compress)
      compress="${2-}"
      shift
      ;;
    -i | --getindex)
      compress="${2-}"
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  #args=("$@")

  # check required params and arguments
  [[ -z "${dir-}" ]] && die "Missing required parameter: dir"

  return 0
}

parse_params "$@"
setup_colors

# script logic here
msg "${BLUE}Read parameters:${NOFORMAT}"
msg "- directory: ${dir-}"
msg "- output: ${output-}"
msg "- compress: ${compress-}"


# download kallisto index file if necessary
if [ "${compress-}" = true ]; then
  if [ ! -f "$script_dir/human_index.idx" ]; then
    msg "${BLUE}Downloading Kallisto index file. This process may take awhile (2 GB). ${NOFORMAT}"
    wget "https://s3.amazonaws.com/mssm-seq-index/human_index.idx" --directory-prefix "$script_dir"
  else
    msg "${RED}Kallisto index file already present. Skipping download. ${NOFORMAT}"
  fi
fi

mkdir -p "${output-}"

for file in "${dir-}/"*; do
    if [ -d "$file" ]; then

        filename=$(basename "$file")

        msg "---------------------------------------------------"
        msg "---------------------------------------------------"
        msg "${BLUE} Start sample $filename ${NOFORMAT}"
        msg "---------------------------------------------------"
        msg "---------------------------------------------------"

        # Unzip fastq files if necessary
        for f in "$file"/*.fastq*; do
          if ( gzip -t "$f" ) ; then
            gunzip "$f"
          fi
        done

        # Fastqc
        source "$script_dir"/fastqc.sh

        # Kallisto
        source "$script_dir"/kallisto.sh

        # Map to genes
        Rscript "$script_dir"/mapgenes.R "${output-}/Quality Assessment/kallisto/$filename" "$script_dir/human_mapping.rda"

        # Zip fastq
        if ( "${compress-}" = true ) ; then
          echo
          msg "Zipping $filename"/
          msg "-----------------------------------"
          gzip "$file"/*.fastq
        fi
    fi
done

# Aggregate all the output files
Rscript "$script_dir"/consolidatesamples.R "${output-}/Quality Assessment/kallisto" "${output-}"

# parentdir="$(dirname "$dir")"
