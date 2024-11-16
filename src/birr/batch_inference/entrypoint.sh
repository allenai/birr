#!/bin/bash

export NUM_GPUS=$BEAKER_ASSIGNED_GPU_COUNT

if [[ -n $NUM_GPUS ]]; then
  export GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv | tail -1)
else
  export GPU_TYPE=""
fi

sed -i "s/^api_key\: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX/api_key\: ${DD_API_KEY}/g" /etc/datadog-agent/datadog.yaml
sed -i "s/^# hostname\: <HOSTNAME_NAME>/hostname\:/g" /etc/datadog-agent/datadog.yaml
sed -i "s/^# hostname_trust_uts_namespace\: false/hostname_trust_uts_namespace\: true/g" /etc/datadog-agent/datadog.yaml
sed -i "s/^# tags\:/tags:\n  - service:ai2-batch-inference\n  - beaker-node:${BEAKER_NODE_HOSTNAME}\n  - beaker-job-name:${BEAKER_JOB_NAME}\n  - workspace:${WORKSPACE}\n  - owner:${OWNER}\n  - dummy-mode:${DUMMY_MODE}\n  - num-gpus:${NUM_GPUS}\n  - gpu-type:${GPU_TYPE}\n  - beaker-workload-id:${BEAKER_WORKLOAD_ID}\n  - beaker-job-id:${BEAKER_JOB_ID}/g" /etc/datadog-agent/datadog.yaml

mv /etc/datadog-agent/conf.d/ray.d/conf.yaml.example /etc/datadog-agent/conf.d/ray.d/conf.yaml && > /etc/datadog-agent/conf.d/ray.d/conf.yaml
printf "instances:\n  - openmetrics_endpoint: http://localhost:8080" > /etc/datadog-agent/conf.d/ray.d/conf.yaml

service datadog-agent start

nohup python3 -m birr.batch_inference.gpumon &

exec python src/birr/batch_inference/runner.py
