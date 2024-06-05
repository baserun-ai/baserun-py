from baserun.integrations.llamaindex import LLamaIndexInstrumentation
from tests.conftest import get_queued_objects

try:
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.base.response.schema import RESPONSE_TYPE
except ImportError:
    pass


def llama() -> RESPONSE_TYPE:
    LLamaIndexInstrumentation.start()
    documents = SimpleDirectoryReader("tests/test_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine.query(
        "I have flour, sugar and butter. What am I missing if I want to bake oatmeal cookies from my recipe?"
    )


def test_llama_simple():
    llama()

    queued_requests = get_queued_objects()

    assert len(queued_requests) == 2

    llama_request = queued_requests[-1]
    assert llama_request["endpoint"] == "traces"

    tags = llama_request["data"]["tags"]
    assert len(tags) == 2

    query = tags[0]
    assert query["key"] == "Query for nodes retrieval"
    assert (
        "I have flour, sugar and butter. What am I missing if I want to bake oatmeal cookies from my recipe?"
        in query["value"]
    )

    selected_nodes = tags[1]
    assert selected_nodes["key"] == "Selected nodes"
    assert "test_data/oatmeal_cookies" in selected_nodes["value"]
