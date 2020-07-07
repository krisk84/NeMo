#!/bin/bash

buildctl build \
    --frontend=dockerfile.v0 \
    --local context=.. \
    --progress plain \
    --local dockerfile=. \
    --opt target=nemo \
    --output type=docker,name=nemo-stable:0.10.1r2 | docker load
