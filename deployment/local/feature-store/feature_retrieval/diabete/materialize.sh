#!/bin/bash
cd ../../feature_repos/diabete
CURRENT_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME