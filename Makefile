ifdef TAG_SUFFIX
	OPTIONAL_TAG = -$(TAG_SUFFIX)
endif

BIRR_IMAGE_TAG=birr-public$(OPTIONAL_TAG)

# TC does not support TTY operations,
# so this provides and optional param to
# remove `-it` from all the docker run
# commands, while preserving it for local
# runs of individual make invocations.
ifneq ($(TTY_DEVICE),0)
	OPTIONAL_IT = -it
endif

ifdef MOUNT_DIR
	OPTIONAL_MOUNT=-v $(shell pwd)/$(MOUNT_DIR):/artifacts:ro
endif

CUDA_VERSION="12.1.0"
INFERENCE_ENV_FILE = src/birr/batch_inference/docker.env
DOCKER_RUN_INF = docker run --rm --platform linux/amd64 $(OPTIONAL_IT) \
	--entrypoint /bin/bash \
	--env-file $(INFERENCE_ENV_FILE) \
	$(OPTIONAL_MOUNT) \
	$(BIRR_IMAGE_TAG)


run:
	${DOCKER_RUN_INF} -c "bash src/birr/batch_inference/entrypoint.sh"

run-dummy:
	DUMMY_MODE=1 MOUNT_DIR=dummy_mode/dummy_model make run

build:
	docker build \
	    --build-arg CUDA_VERSION=${CUDA_VERSION} \
	    --platform linux/amd64 \
	    -t $(BIRR_IMAGE_TAG) . \
	    -f src/birr/batch_inference/Dockerfile

mypy:
	@${DOCKER_RUN_INF} -c 'mypy src/birr/ tests/'

check-format:
	@${DOCKER_RUN_INF} -c 'black --check --diff src/birr tests'

format:
	docker run --rm --platform linux/amd64 $(OPTIONAL_IT) \
	     --entrypoint /bin/bash \
	     -v $(shell pwd):/work:rw \
	     $(BIRR_IMAGE_TAG) \
	     -c 'black /work/src/birr /work/tests'

test:
	@${DOCKER_RUN_INF} -c 'pytest --log-level=INFO tests/birr/'

