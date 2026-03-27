from langchain_community.document_loaders import DirectoryLoader, UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator


def test_data_factory():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    langchain_llm = LangchainLLMWrapper(llm)
    embed  = OpenAIEmbeddings()
    loader = DirectoryLoader(
        path="tests/testdata",
        glob="**/*.docx",
        loader_cls=UnstructuredWordDocumentLoader
    )
    docs = loader.load()
    generate_embedding = LangchainEmbeddingsWrapper(embed)
    generator = TestsetGenerator(llm=langchain_llm, embedding_model=generate_embedding)
    datasets = generator.generate_with_langchain_docs(docs, testset_size=20)
    print(datasets.to_list())