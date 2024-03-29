ARG AWS_ACCOUNT_ID

FROM ${AWS_ACCOUNT_ID}.dkr.ecr.eu-west-1.amazonaws.com/org.wellcome/ml-services:tornado-latest
# This docker assumes that models will mounted in a folder called mnt/vol/models

WORKDIR /code

COPY requirements_minimal.txt .

RUN pip install -r requirements_minimal.txt

# setup.py needs README.md
COPY nutrition_labels nutrition_labels/
COPY README.md setup.py .
RUN pip install -e . --no-deps 

COPY api api/
COPY configs configs/

ENV MODELS_PATH=/mnt/vol/models

CMD ["python", "api/entrypoint.py"]
