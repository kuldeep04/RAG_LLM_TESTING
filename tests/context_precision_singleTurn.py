import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference


# user_input = query
# response = LLM Response
# Reference = what is ground truth ( how many reference present )
# Retrived context = top k documents retrieved documents


def test_context_precision(sync_llm, getdata):
    # create object of class metric
    # LLM + method metric = score
    context_precision = LLMContextPrecisionWithoutReference(llm=sync_llm)

    # Feed data - refer to class fixture.

    # score
    score = context_precision.single_turn_score(getdata)
    print(score)
    assert score > 0.9

@pytest.fixture
def getdata():
    question = "How many article are there in the selenium webdriver python course?"
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": question,
                                     "chat_history": []
                                 }).json()
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"],
                            responseDict["retrieved_docs"][3]["page_content"]],
        response=responseDict["answer"]
    )
    return sample