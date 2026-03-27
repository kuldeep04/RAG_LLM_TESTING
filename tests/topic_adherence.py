import pytest
from ragas import MultiTurnSample
from ragas.metrics.collections import TopicAdherence
from ragas.messages import HumanMessage, AIMessage

@pytest.mark.asyncio
async def test_topicAdherence(get_async_llm, getdata):
    topicAdherence = TopicAdherence(llm=get_async_llm, mode="f1")
    score = await topicAdherence.ascore(getdata.user_input, getdata.reference_topics)
    print(score)

    assert score is not None
    assert score > 0.47

@pytest.fixture
def getdata():

    conversation = [
        HumanMessage(content="How many article are there in the selenium webdriver python course?"),
        AIMessage(content="There are 23 articles included in the Selenium WebDriver Python course"),
        HumanMessage(content="How many downloadable resources are there in the selenium webdriver python course?"),
        AIMessage(content="There are 9 downloadable resources in the Selenium WebDriver Python course."),
    ]

    reference = [""" 
    Then AI should:
    1. Give result related to the selenium webdriver python course
    2. there are 23 articles in the selenium webdriver python course
    """]
    return MultiTurnSample(
        user_input=conversation,
        reference_topics=reference
    )