import pytest
from ragas import SingleTurnSample,  experiment
from ragas.metrics._answer_relevance import ResponseRelevancy
from ragas.metrics.collections import FactualCorrectness

from tests.test_utils import get_llm_response, load_test_data

@pytest.mark.asyncio
@pytest.mark.parametrize("getdata", load_test_data("multiple_matrics.json"), indirect=True)
async def test_relevancy_factual( get_async_llm,get_embeddings, getdata):
    relevancy = ResponseRelevancy(llm=get_async_llm, embeddings=get_embeddings)
    factual = FactualCorrectness(llm=get_async_llm)

    score1 = relevancy.single_turn_score(getdata)

    score2 = await factual.ascore(getdata.response, getdata.reference)

    print(f"relevancy: {score1.to}, factual: {score2}")

    assert score1 is not None
    assert score2 is not None


def build_experiment(eval_dataset, metrics, llm):
    @experiment
    def run_eval():
        return {
            "dataset": eval_dataset,
            "metrics": metrics,
            "llm": llm
        }

    return run_eval


@pytest.fixture
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)

    if not responseDict:
        raise ValueError("LLM response is None")

    return SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[doc["page_content"] for doc in responseDict.get("retrieved_docs")],
        response=responseDict["answer"],
        reference=test_data["reference"]
    )
