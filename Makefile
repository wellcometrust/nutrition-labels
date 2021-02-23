
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OSFLAG := linux
endif
ifeq ($(UNAME_S),Darwin)
	OSFLAG := macosx_10_13
endif


PRODIGY_BUCKET := datalabs-packages/Prodigy
PROJECT_BUCKET := datalabs-public/nutrition_labels

VIRTUALENV := build/virtualenv
PRODIGY_VIRTUALENV := build/prodigy_virtualenv
PRODIGY_WHEEL := prodigy-1.8.5-cp35.cp36.cp37-cp35m.cp36m.cp37m-$(OSFLAG)_x86_64.whl

$(VIRTUALENV)/.installed: requirements.txt
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip3 install -r requirements.txt
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed

# Prodigy Virtual Environment (since Prodigy requires a different Spacy version)

$(PRODIGY_VIRTUALENV)/.installed: prodigy_requirements.txt
	@if [ -d $(PRODIGY_VIRTUALENV) ]; then rm -rf $(PRODIGY_VIRTUALENV); fi
	@mkdir -p $(PRODIGY_VIRTUALENV)
	virtualenv --python python3 $(PRODIGY_VIRTUALENV)
	$(PRODIGY_VIRTUALENV)/bin/pip3 install -r prodigy_requirements.txt
	touch $@

$(PRODIGY_VIRTUALENV)/.en_core_web_sm: 
	$(PRODIGY_VIRTUALENV)/bin/python -m spacy download en_core_web_sm
	touch $@

.PHONY: $(PRODIGY_WHEEL)
$(PRODIGY_WHEEL): 
	aws s3 cp s3://$(PRODIGY_BUCKET)/$(PRODIGY_WHEEL) ./build/$(PRODIGY_WHEEL)
	$(PRODIGY_VIRTUALENV)/bin/pip3 install ./build/$(PRODIGY_WHEEL)

.PHONY: prodigy_virtualenv
prodigy_virtualenv: $(PRODIGY_VIRTUALENV)/.installed $(PRODIGY_WHEEL) $(PRODIGY_VIRTUALENV)/.en_core_web_sm

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

.PHONY: test
test:
	$(VIRTUALENV)/bin/pytest --tb=line ./tests
