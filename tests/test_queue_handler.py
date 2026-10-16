from minio_handler import MinioHandler
from queue_handler import QueueHandler


def test_search_queue_names():
    q = QueueHandler()
    assert q.request_queue == "AI_SEARCH_QUEUE_IN"
    assert "AI_SEARCH_EXCHANGE_INPUT" in q.request_exchanges
    assert q.request_exchange in q.request_exchanges
    assert q.result_queue == "AI_SEARCH_QUEUE_OUT"
    assert q.result_exchange in q.result_exchanges
    assert q.workflow_type == "process_search"
    assert q.dlq_routing_key == "search.dlq"


def test_minio_object_name_from_url():
    m = MinioHandler.__new__(MinioHandler)
    m.bucket_name = "ai-inhouse"
    assert m.object_name("search/req1.jpg") == "search/req1.jpg"
    assert m.object_name("https://minio.example/ai-inhouse/search/req1.jpg") == "search/req1.jpg"
    assert m.object_name("") == ""
