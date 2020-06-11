PROJECT_BUCKET := datalabs-public/nutrition_labels

VIRTUALENV := build/virtualenv

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

# Sync data from S3

.PHONY:sync_data_to_s3
sync_data_to_s3:
	aws s3 sync data/ s3://$(PROJECT_BUCKET)/data/

.PHONY:sync_data_from_s3
sync_data_from_s3:
	aws s3 sync s3://$(PROJECT_BUCKET)/data/ data/

.PHONY:sync_models_to_s3
sync_models_to_s3:
	aws s3 sync models/ s3://$(PROJECT_BUCKET)/models/

.PHONY:sync_models_from_s3
sync_models_from_s3:
	aws s3 sync s3://$(PROJECT_BUCKET)/models/ models/
