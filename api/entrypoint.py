import json
import logging
from typing import Any, Dict, Union, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic.schema import schema
from tornado.web import RequestHandler

from ml_service import APIService, define


# Modify with relevant information about input data
class DataPoint(BaseModel):
    text: str
    synopsis: str
    metadata: Optional[Dict[str, Any]]


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
with open('api_doc_base.yaml', 'r') as f:
    api_docs = yaml.load(f, Loader=yaml.FullLoader)

top_level_schema = schema([DataPoint, Prediction], ref_prefix='#/components/schemas/')
api_docs['components'] = {'schemas': top_level_schema['definitions']}


# Extends the tornado RequestHandler
class MLEndpoint(RequestHandler):
    async def post(self):
        # Coerce the JSON POST request body to your data model
        # type
        data = DataPoint(**json.loads(self.request.body))

        # Create your result
        result = Prediction(predicted_class="Tech", predicted_proba=0.1)

        # Write the output, this is an async handler
        self.write(result.json())

        # This is an async handler, so you can carry on and do other things
        # after having written out the response data to the socket above,
        # for instance record some metrics with `self.record_metric()`
        # output some logs or otherwise
    async def get(self):
        self.write({
            "model_name": api_docs["info"]["title"],
            "version": api_docs["info"]["version"]
        })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = APIService(
        handler_class=MLEndpoint,
        doc_json=api_docs,
        ml_name="Nutrition Labels",
    )

    app.run_forever()


