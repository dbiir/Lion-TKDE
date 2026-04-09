#!/bin/bash

set -e

BUILD_DIR="build"

mkdir -p "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 
# -DCMAKE_CXX_FLAGS=-pg
cmake --build "$BUILD_DIR" -j8

# ./sync.bash
