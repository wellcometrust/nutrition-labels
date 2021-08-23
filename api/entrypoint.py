import configparser
import json
import logging
import os
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic.schema import schema
from tornado.web import RequestHandler

from ml_service import APIService, define

from nutrition_labels.grant_tagger import GrantTagger

logging.basicConfig()
logging.root.setLevel(level=logging.INFO)

here = os.path.abspath(os.path.dirname(__file__))
models_path = os.getenv('MODELS_PATH', '/mnt/vol/models')
model_version = os.getenv('MODEL_VERSION', '2021.07.0')


# Modify with relevant information about input data
class DataPoint(BaseModel):
    title: str
    description: str
    metadata: Optional[Dict[str, Any]] = Field(None, description="Any extra text")


# Modify with relevant information returned by the prediction endpoint
class Prediction(BaseModel):
    predicted_class: str = Field(..., description="The class")
    predicted_proba: float = Field(..., description="Prob between 0 and 1")

    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "Tech",
                "predicted_proba": 0.1,
            }
        }


# Generates docs
with open(os.path.join(here, 'api_doc_base.yaml'), 'r') as f:
    api_docs = yaml.load(f, Loader=yaml.FullLoader)

top_level_schema = schema([DataPoint, Prediction], ref_prefix='#/components/schemas/')
api_docs['components'] = {'schemas': top_level_schema['definitions']}

logging.info("Loading model")

config = configparser.ConfigParser()
config.read(os.path.join(here, f'../configs/predict/{model_version}.ini'))

grant_tagger = GrantTagger(
    threshold=config.getfloat('model_parameters', 'pred_prob_thresh'),
    prediction_cols=config['prediction_data']['grant_text_cols'].split(',')
)

grant_tagger.load_model(os.path.join(models_path, config['model_parameters']['model_dirs']))


# Modify with relevant prediction code
# Extends the tornado RequestHandler
class MLEndpoint(RequestHandler):
    async def post(self):
        # Coerce the JSON POST request body to your data model
        # type

        # I want to allow request_body to be a list of texts too for batch predict
        request_body = json.loads(self.request.body)

        if isinstance(request_body, dict):
            request_body = [request_body]

        data = [DataPoint(**point).dict() for point in request_body]

        # Generate the prediction
        result = []
        X_vec = grant_tagger.transform(data)
        for y_pred in grant_tagger.predict_proba(X_vec):

            class_map = {0: "No Tech", 1: "Tech"}

            class_predicted = y_pred.argmax()
            prob = y_pred[class_predicted]

            result += [
                Prediction(predicted_class=class_map[class_predicted], predicted_proba=prob).dict()
            ]

        # Write the output, this is an async handler
        self.write({"result": result})

        # This is an async handler, so you can carry on and do other things
        # after having written out the response data to the socket above,
        # for instance record some metrics with `self.record_metric()`
        # output some logs or otherwise
    async def get(self):
        self.write({
            "model_name": api_docs["info"]["title"],
            "version": model_version
        })


if __name__ == "__main__":

    app = APIService(
        handler_class=MLEndpoint,
        doc_json=api_docs,
        ml_name=api_docs["info"]["title"],
        ml_version=model_version
    )

    app.run_forever()
