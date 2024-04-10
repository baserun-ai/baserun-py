from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.base.response.schema import RESPONSE_TYPE

from baserun import Baserun


@Baserun.trace
def llama() -> RESPONSE_TYPE:
    documents = SimpleDirectoryReader("test_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine.query(
        "I have flour, sugar and butter. What am I missing if I want to bake oatmeal cookies from my recipe?"
    )


def test_llama_simple(mock_services):
    response = llama()

    submit_log_calls = mock_services["submission_service"].SubmitLog.method_calls
    assert len(submit_log_calls) == 2
    assert submit_log_calls[0].args[0].log.name == "Query for nodes retrieval"
    assert submit_log_calls[1].args[0].log.name == "Selected nodes"
    spans = list(sr.span for sr in Baserun.exporter_queue.queue)
    # documents embeddings + query embeddings + completion. using this small test data should be 1 of each
    assert len(spans) == 3
    assert spans[0].request_type == "embeddings"
    assert spans[1].request_type == "embeddings"
    assert spans[2].request_type == "chat"
    assert spans[2].completions[0].content == response.response
