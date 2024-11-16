#!/usr/bin/env bash

FAILED=0
FAILURE_REASONS=()

export TTY_DEVICE=0
export TAG_SUFFIX="${BUILD_NUMBER:-$(date +%s)}"

add_failure () {
  FAILED=1
  FAILURE_REASONS+=("Failed $1. Investigate locally with: $2")
}

make build || add_failure "image build" "make build"
make check-format || add_failure "formatting" "make check-format"
make mypy || add_failure "type checking" "make mypy"
make test || add_failure "tests failed" "make test"

for failure_reason in "${FAILURE_REASONS[@]}"
do
  echo "$failure_reason" >&2
done

exit $FAILED
