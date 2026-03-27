import pytest
from ragas import SingleTurnSample
from ragas.metrics.collections import DomainSpecificRubrics


@pytest.mark.asyncio
async def test_rubricsScore(get_async_llm, test_data):
    custom_rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
        "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
    }
    rubrics_score = DomainSpecificRubrics(llm=get_async_llm, rubrics=custom_rubrics, with_reference=True)
    score = await rubrics_score.ascore(user_input=test_data.user_input,
                                       response=test_data.response,
                                       reference=test_data.reference)

    print(f"Score: {score.value}, Feedback: {score.reason}")


@pytest.fixture
def test_data():
    return SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Europe and it is part of France.",
        reference="The Eiffel Tower is located in Paris."
    )
