#!/usr/bin/env bash
# ==============================================================================
# PyTorch Installation Script for ESPnet
# ==============================================================================
#
# Description:
#   This script installs PyTorch and TorchAudio with appropriate CUDA versions.
#   It supports both conda and pip installation methods and handles various
#   PyTorch versions with their corresponding CUDA compatibility.
#
# Usage:
#   ./install_torch.sh <use_conda> <torch_version> <cuda_version>
#
# Arguments:
#   use_conda      - Installation method: "true" for conda, "false" for pip
#   torch_version  - PyTorch version to install (e.g., "2.0.0", "1.13.0")
#   cuda_version   - CUDA version (e.g., "11.8", "12.1") or "cpu" for CPU-only
#
# Examples:
#   ./install_torch.sh true 2.0.0 11.8    # Install PyTorch 2.0.0 with CUDA 11.8 via conda
#   ./install_torch.sh false 2.1.0 cpu    # Install PyTorch 2.1.0 CPU-only via pip
#
# Notes:
#   - CUDA is not supported on macOS
#   - PyTorch >= 2.6.0 automatically falls back to pip installation
#   - The script validates Python and CUDA version compatibility
#   - Unsupported CUDA versions will automatically fallback to the nearest supported version
#
# Refactoring History:
#   - Extracted OS detection into detect_os() function
#   - Created validate_arguments() for input validation
#   - Created normalize_cuda_version() for CUDA version processing
#   - Created install_packaging() to ensure packaging module is available
#   - Created python_plus() and pytorch_plus() for version comparison
#   - Created install_torch() for the actual installation logic
#   - Created check_python_version() for Python compatibility validation
#   - Created check_cuda_version() for CUDA compatibility validation
#   - Created get_torchaudio_version() to map PyTorch to TorchAudio versions
#   - Created main installation dispatcher install_pytorch_by_version()
#   - Improved error messages and logging throughout
#
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# Logging function
# ------------------------------------------------------------------------------
# Logs messages with timestamp, file location, and function name
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# ------------------------------------------------------------------------------
# OS Detection
# ------------------------------------------------------------------------------
# Detects the operating system type (linux, macos, windows, or unknown)
# Sets the global variable 'os_type'
detect_os() {
    local unames="$(uname -s)"
    if [[ ${unames} =~ Linux ]]; then
        os_type=linux
    elif [[ ${unames} =~ Darwin ]]; then
        os_type=macos
    elif [[ ${unames} =~ MINGW || ${unames} =~ CYGWIN || ${unames} =~ MSYS ]]; then
        os_type=windows
    else
        os_type=unknown
    fi
    log "[INFO] Detected OS: ${os_type}"
}

# ------------------------------------------------------------------------------
# Argument Validation
# ------------------------------------------------------------------------------
# Validates command-line arguments and sets global variables
# Arguments:
#   $1 - use_conda (true/false)
#   $2 - torch_version
#   $3 - cuda_version
validate_arguments() {
    if [ $# -ne 3 ]; then
        log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
        exit 1
    fi

    use_conda="$1"
    if [ "${use_conda}" != false ] && [ "${use_conda}" != true ]; then
        log "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
        exit 1
    fi

    torch_version="$2"
    cuda_version="$3"
}

# ------------------------------------------------------------------------------
# CUDA Version Normalization
# ------------------------------------------------------------------------------
# Normalizes and validates CUDA version
# - Converts "cpu" or "CPU" to empty string
# - Validates CUDA is not used on macOS
# - Sets pip_cpu_module_suffix based on OS
normalize_cuda_version() {
    if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
        cuda_version=
    fi

    if [ -n "${cuda_version}" ] && [ "${os_type}" = macos ]; then
        log "[ERROR] cuda is not supported for MacOS"
        exit 1
    fi

    if [ "${os_type}" == macos ]; then
        pip_cpu_module_suffix=
    else
        pip_cpu_module_suffix="+cpu"
    fi

    cuda_version_without_dot="${cuda_version/\./}"
}

# ------------------------------------------------------------------------------
# Environment Setup
# ------------------------------------------------------------------------------
# Retrieves Python version and ensures packaging module is installed
setup_environment() {
    python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
    log "[INFO] python_version=${python_version}"
}

# ------------------------------------------------------------------------------
# Install Packaging Module
# ------------------------------------------------------------------------------
# Ensures the Python packaging module is available for version comparisons
install_packaging() {
    if ! python -c "import packaging.version" &> /dev/null; then
        log "[INFO] Installing packaging module"
        python3 -m pip install packaging
    fi
}

# ------------------------------------------------------------------------------
# Version Comparison Functions
# ------------------------------------------------------------------------------
# Compares Python version with a given version
# Arguments:
#   $1 - version to compare against
# Returns:
#   "true" if current Python version >= given version, "false" otherwise
python_plus() {
    python3 <<EOF
from packaging.version import parse as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

# Compares PyTorch version with a given version
# Arguments:
#   $1 - version to compare against
# Returns:
#   "true" if target PyTorch version >= given version, "false" otherwise
pytorch_plus() {
    python3 <<EOF
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

# ------------------------------------------------------------------------------
# PyTorch Installation
# ------------------------------------------------------------------------------
# Installs PyTorch and TorchAudio using either conda or pip
# Arguments:
#   $1 - torchaudio version to install
# Global variables used:
#   use_conda, torch_version, cuda_version, cuda_version_without_dot
install_torch() {
    local torchaudio_version="$1"

    # PyTorch >= 2.6.0 doesn't support conda installation reliably
    if $(pytorch_plus 2.6.0) && [ "${use_conda}" = true ]; then
        log "[INFO] PyTorch >= 2.6.0: Fallback use_conda: true -> false"
        use_conda=false
    fi

    if "${use_conda}"; then
        install_torch_conda "${torchaudio_version}"
    else
        install_torch_pip "${torchaudio_version}"
    fi
}

# ------------------------------------------------------------------------------
# Conda Installation Method
# ------------------------------------------------------------------------------
# Installs PyTorch and TorchAudio using conda
# Arguments:
#   $1 - torchaudio version
install_torch_conda() {
    local torchaudio_version="$1"

    if $(pytorch_plus 1.13.0); then
        # PyTorch 1.13+ uses pytorch-cuda instead of cudatoolkit
        if [ -z "${cuda_version}" ]; then
            log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" cpuonly -c pytorch
            conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" cpuonly -c pytorch
        else
            log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "pytorch-cuda=${cuda_version}" -c pytorch -c nvidia
            conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "pytorch-cuda=${cuda_version}" -c pytorch -c nvidia
        fi
    else
        # PyTorch < 1.13 uses cudatoolkit
        if [ -z "${cuda_version}" ]; then
            log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" cpuonly -c pytorch
            conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" cpuonly -c pytorch
        else
            install_torch_conda_with_cudatoolkit "${torchaudio_version}"
        fi
    fi
}

# ------------------------------------------------------------------------------
# Conda Installation with cudatoolkit
# ------------------------------------------------------------------------------
# Handles special cases for cudatoolkit installation via conda
# Arguments:
#   $1 - torchaudio version
install_torch_conda_with_cudatoolkit() {
    local torchaudio_version="$1"
    local cudatoolkit_channel

    if [ "${cuda_version}" = "11.5" ] || [ "${cuda_version}" = "11.6" ]; then
        # CUDA 11.5/11.6 requires conda-forge channel
        cudatoolkit_channel=conda-forge
        log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
        conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
    elif [ "${cuda_version}" = "11.1" ] || [ "${cuda_version}" = "11.2" ]; then
        # CUDA 11.1/11.2 requires nvidia channel
        cudatoolkit_channel=nvidia
        log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
        conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
    else
        # Standard conda installation
        log conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch
        conda install -y "pytorch=${torch_version}" "torchaudio=${torchaudio_version}" "cudatoolkit=${cuda_version}" -c pytorch
    fi
}

# ------------------------------------------------------------------------------
# Pip Installation Method
# ------------------------------------------------------------------------------
# Installs PyTorch and TorchAudio using pip
# Arguments:
#   $1 - torchaudio version
install_torch_pip() {
    local torchaudio_version="$1"

    if [ -z "${cuda_version}" ]; then
        log python3 -m pip install "torch==${torch_version}" "torchaudio==${torchaudio_version}" --extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install "torch==${torch_version}" "torchaudio==${torchaudio_version}" --extra-index-url https://download.pytorch.org/whl/cpu
    else
        log python3 -m pip install "torch==${torch_version}" "torchaudio==${torchaudio_version}" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
        python3 -m pip install "torch==${torch_version}" "torchaudio==${torchaudio_version}" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
    fi
}

# ------------------------------------------------------------------------------
# Python Version Validation
# ------------------------------------------------------------------------------
# Checks if current Python version is compatible with the target PyTorch version
# Arguments:
#   $1 - maximum supported Python version
# Exits with error if Python version is incompatible
check_python_version() {
    local max_version="$1"
    local min_version

    if $(pytorch_plus 2.1.0); then
        min_version=3.7
    else
        min_version=3.6
    fi
    
    if $(python_plus "${max_version}") || ! $(python_plus "${min_version}"); then
        log "[ERROR] pytorch=${torch_version} requires Python >=${max_version}, <=${min_version}, but your Python is ${python_version}"
        exit 1
    fi
}

# ------------------------------------------------------------------------------
# CUDA Version Validation
# ------------------------------------------------------------------------------
# Checks if the requested CUDA version is supported by the PyTorch version
# Arguments:
#   $@ - list of supported CUDA versions
# Returns:
#   0 if supported, 1 if not supported
check_cuda_version() {
    local supported=false
    local v

    for v in "" "$@"; do
        [ "${cuda_version}" = "${v}" ] && supported=true
    done

    if ! "${supported}"; then
        log "[WARNING] Pre-built package for PyTorch=${torch_version} with CUDA=${cuda_version} is not provided."
        return 1
    fi
    return 0
}

# ------------------------------------------------------------------------------
# CUDA Version Fallback Helper
# ------------------------------------------------------------------------------
# Adjusts CUDA version to the nearest supported version if current is unsupported
# Arguments:
#   $1 - fallback CUDA version
#   $@ - list of supported CUDA versions (passed to check_cuda_version)
set_cuda_fallback() {
    local fallback_version="$1"
    shift

    if ! check_cuda_version "$@"; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=${fallback_version}"
        cuda_version="${fallback_version}"
        cuda_version_without_dot="${cuda_version/\./}"
    fi
}

# PyTorch version configuration table
# Format: py_max:cuda_versions:torchaudio_ver:fallback_cuda
declare -A PYTORCH_CONFIG=(
    [2.9.0]="3.13:cuda:12.6,12.8,13.0:2.9.0:12.8"
    [2.8.0]="3.13:cuda:12.6,12.8,12.9:2.8.0:12.8"
    [2.7.1]="3.13:cuda:11.8,12.6,12.8:2.7.1:12.6"
    [2.6.0]="3.13:cuda:12.6,12.4,11.8:2.6.0:12.6"
    [2.5.1]="3.13:cuda:12.4,12.1,11.8:2.5.1:12.4"
    [2.5.0]="3.13:cuda:12.4,12.1,11.8:2.5.0:12.4"
    [2.4.1]="3.13:cuda:12.4,12.1,11.8:2.4.1:12.4"
    [2.4.0]="3.13:cuda:12.4,12.1,11.8:2.4.0:12.4"
    [2.3.1]="3.13:cuda:12.1,11.8:2.3.1:12.1"
    [2.3.0]="3.13:cuda:12.1,11.8:2.3.0:12.1"
    [2.2.2]="3.13:cuda:12.1,11.8:2.2.2:12.1"
    [2.2.1]="3.13:cuda:12.1,11.8:2.2.1:12.1"
    [2.2.0]="3.13:cuda:12.1,11.8:2.2.0:12.1"
    [2.1.2]="3.12:cuda:12.1,11.8:2.1.2:12.1"
    [2.1.1]="3.12:cuda:12.1,11.8:2.1.1:12.1"
    [2.1.0]="3.12:cuda:12.1,11.8:2.1.0:12.1"
    [2.0.1]="3.12:cuda:11.8,11.7:2.0.2:11.8"
    [2.0.0]="3.12:cuda:11.8,11.7:2.0.0:11.8"
    [1.13.1]="3.11:cuda:11.7,11.6:0.13.1:11.7"
    [1.13.0]="3.11:cuda:11.7,11.6:0.13.0:11.7"
    [1.12.1]="3.11:cuda:11.6,11.3,10.2:0.12.1:11.6"
    [1.12.0]="3.11:cuda:11.6,11.3,10.2:0.12.0:11.6"
    [1.11.0]="3.11:cuda:11.5,11.3,11.1,10.2:0.11.0:11.5"
    [1.10.2]="3.10:cuda:11.3,11.1,10.2:0.10.2:11.3"
    [1.10.1]="3.10:cuda:11.3,11.1,10.2:0.10.1:11.3"
    [1.10.0]="3.11:cuda:11.3,11.1,10.2:0.10.0:11.3"
    [1.9.1]="3.10:cuda:11.1,10.2:0.9.1:11.7"
    [1.9.0]="3.10:cuda:11.1,10.2:0.9.0:11.1"
    [1.8.1]="3.10:cuda:11.1,10.2,10.1:0.8.1:11.1"
    [1.8.0]="3.10:cuda:11.1,10.2,10.1:0.8.0:11.1"
    [1.7.1]="3.10:cuda:11.0,10.2,10.1,9.2:0.7.2:11.0"
    [1.7.0]="3.10:cuda:11.0,10.2,10.1,9.2:0.7.0:11.0"
    [1.6.0]="3.9:cuda:10.2,10.1,9.2:0.6.0:10.2"
    [1.5.1]="3.9:cuda:10.2,10.1,9.2:0.5.1:10.2"
    [1.5.0]="3.9:cuda:10.2,10.1,9.2:0.5.0:10.2"
    [1.4.0]="3.9:cuda:10.1,10.0,9.2:0.4.0:10.1"
    [1.3.1]="3.8:cuda:10.1,10.0,9.2:0.3.2:10.1"
    [1.3.0]="3.8:cuda:10.1,10.0,9.2:0.3.1:10.1"
    [1.2.0]="3.8:cuda:10.0,9.2:0.3.0:10.0"
)

# ------------------------------------------------------------------------------
# PyTorch Version-Specific Installation Dispatcher
# ------------------------------------------------------------------------------
# Main installation dispatcher that handles version-specific logic
# This function contains the mapping between PyTorch versions, compatible
# Python versions, CUDA versions, and corresponding TorchAudio versions
install_pytorch_by_version() {
    if [[ ${PYTORCH_CONFIG[${torch_version}]+_} ]]; then
        log "[INFO] torch_version=${torch_version}"
        log "[INFO] cuda_version=${cuda_version}"

        IFS=':' read -r py_max _ cuda_versions torchaudio_ver fallback_cuda <<< "${PYTORCH_CONFIG[${torch_version}]}"

        check_python_version "${py_max}"
        set_cuda_fallback "${fallback_cuda}" ${cuda_versions//,/ }

        install_torch "${torchaudio_ver}"

    # Unsupported versions
    else
        log "[ERROR] This script doesn't support pytorch=${torch_version},\n"\
            "or the version does not exist."
        exit 1
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================
# Script entry point - orchestrates the installation process
main() {
    # Step 1: Detect operating system
    detect_os

    # Step 2: Validate and parse command-line arguments
    validate_arguments "$@"

    # Step 3: Normalize CUDA version and set environment variables
    normalize_cuda_version

    # Step 4: Set up Python environment
    setup_environment

    # Step 5: Ensure packaging module is installed
    install_packaging

    # Step 6: Install PyTorch based on version
    install_pytorch_by_version

    log "[INFO] Installation completed successfully"
}

# Execute main function with all script arguments
main "$@"
