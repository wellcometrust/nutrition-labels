#!/usr/bin/env bash

# Do not send slack message if train fails
set -e

curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <$SLACK_USER>, your pipeline has started training'}" $SLACK_HOOK

pipelines/tech_grants_pipeline.sh

# could go to its own script
curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <$SLACK_USER>, your model has finished. :dancingbanana:'}" $SLACK_HOOK
