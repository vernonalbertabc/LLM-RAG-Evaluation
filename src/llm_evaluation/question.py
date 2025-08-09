import uuid

from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.query_engine import BaseQueryEngine
from pydantic import BaseModel, Field


class Question(BaseModel):
    key: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique Identifier of query",
    )
    query: str = Field(..., description="Query to ask.")
    response: RESPONSE_TYPE | None = Field(default=None, description="The response to the query.")
    model_config = {"arbitrary_types_allowed": True}

    def __call__(self, query_engine: BaseQueryEngine) -> RESPONSE_TYPE:
        self.response = query_engine.query(self.query)
        return self.response


class QuestionStack(BaseModel):
    questions: list[Question] = Field(default_factory=list, description="List of Questions")

    def __call__(self, query_engine: BaseQueryEngine) -> list[Question] | None:
        # TODO: You could implement a method to iterate over the questions and call each one.
        pass


# TODO: Implement your own evaluation framework to evaluate the responses.
