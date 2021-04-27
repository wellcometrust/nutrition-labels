#!/usr/bin/env bash

# Do not send slack message if train fails
set -e

curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <@UL32YM1FC>, your pipeline has started training'}" $SLACK_HOOK

./tech_grants_pipeline.sh

# could go to its own script
curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <@UL32YM1FC>, your model has finished. :dancingbanana:'}" $SLACK_HOOK
