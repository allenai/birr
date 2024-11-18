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

## Project Setup

```bash
cd <project_root>
python3 -m venv venv
source venv/bin/activate
pip install .[batch_inference,vllm]
```

## Prepare Your Data

Records you want to run inference over must be partitioned
into one or more jsonl files in a flat directory. Each row
should have the following structure:

```json
{"chat_messages": [{"role":  "user", "content":  "asdf"}]}
```

Additional fields may be provided in each row (e.g. ids, metadata),
and will be preserved in output.

## Define A Job

Author a configuration file for your job, see example file here:

<TODO: LINK TO SAMPLE CONFIG>

## Run Your Job

```bash
# in activated venv
python src/birr/batch_inference/runner.py --config-file <path_to_config_file>
```
