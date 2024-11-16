# birr

Batch InfeRence Runtime

# Overview

A simplified local-only release of the toolchain
used by AI2 to perform large-scale inference
through LLMs and VLMs.

This image orchestrates inference jobs based
on a user-provided YAML config file.

It leverages:

* ray for concurrency management
* vllm for inference backend

It consumes JSONL work files and outputs
result files to a chosen destination.

# Usage

-- TODO
