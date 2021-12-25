#!/bin/sh
#
# Set environment variables.

export IMAGE_NAME=${USER}_explanation
export CONTAINER_NAME=${USER}_explanation

if [ -e docker/env_dev.sh ]; then
  . docker/env_dev.sh
fi