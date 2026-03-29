#!/bin/bash
set -e

PYTHON_VERSIONS=("cp38-cp38" "cp39-cp39" "cp310-cp310" "cp311-cp311" "cp312-cp312" "cp313-cp313")

rm -rf dist build

for py_version in "${PYTHON_VERSIONS[@]}"; do
    echo "====================================="
    echo "Processing Python version: $py_version"
    echo "====================================="

    export PYTHON_PATH="/opt/python/$py_version/bin/"
    export PYTHON_EXE="$PYTHON_PATH/python"
    export PIP_EXE="$PYTHON_PATH/pip"

    echo "Building wheel..."
    $PIP_EXE install setuptools
    $PYTHON_EXE -m build --wheel --no-isolation .

    $PIP_EXE uninstall -y nanoctrl

    echo "Completed processing $py_version"
    echo ""
done

echo "Ready for upload..."

/opt/python/cp312-cp312/bin/twine upload dist/*
