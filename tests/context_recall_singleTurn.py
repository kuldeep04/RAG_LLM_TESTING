
import pytest
from ragas import SingleTurnSample
from ragas.metrics._context_recall import LLMContextRecall
from tests.test_utils import get_llm_response, load_test_data


@pytest.mark.parametrize("getdata", load_test_data("context_recall_singleTurn.json"), indirect=True)
def test_context_recall(sync_llm, getdata):
    context_recall = LLMContextRecall(llm=sync_llm)
    score = context_recall.single_turn_score(getdata)
    print(score)
    assert score > 0.7


@pytest.fixture
def getdata(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)

    sample = SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[responseDict["retrieved_docs"][0]["page_content"],
                            responseDict["retrieved_docs"][1]["page_content"],
                            responseDict["retrieved_docs"][2]["page_content"],
                            responseDict["retrieved_docs"][3]["page_content"]],
        reference=test_data["reference"]
    )
    return sample
