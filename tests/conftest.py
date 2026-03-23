import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper, llm_factory

@pytest.fixture
def get_embeddings():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return LangchainEmbeddingsWrapper(embeddings)

@pytest.fixture
def sync_llm():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )
    return LangchainLLMWrapper(llm)


@pytest.fixture
def get_async_llm():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    client = AsyncOpenAI()
    llm = llm_factory(
        "gpt-4o-mini",  # or gpt-5-mini equivalent
        client=client
    )
    return llm
