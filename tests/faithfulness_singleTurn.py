import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import Faithfulness

from tests.test_utils import load_test_data, get_llm_response

@pytest.mark.asyncio
@pytest.mark.parametrize("getdata", load_test_data("faithful_singleTurn.json"), indirect=True)
async def test_faithfulness_singleTurn(get_async_llm, getdata):
    faithful = Faithfulness(llm=get_async_llm)
    score = await faithful.ascore(
        user_input=getdata.user_input,
        response=getdata.response,
        retrieved_contexts=getdata.retrieved_contexts
    )
    print(score)
    assert score is not None
    assert score > 0.8


@pytest.fixture
def getdata(request) -> SingleTurnSample:
    test_data = request.param
    responseDict = get_llm_response(test_data)
    if not responseDict:
        raise ValueError("LLM response is None")

    retrieved_docs = responseDict.get("retrieved_docs", [])
    if not retrieved_docs:
        raise ValueError(f"No retrieved_docs found: {responseDict}")

    return SingleTurnSample(
        user_input=test_data["question"],
        retrieved_contexts=[
            doc.get("page_content", "") for doc in retrieved_docs[:3]
        ],
        response=responseDict.get("answer", "")
    )
