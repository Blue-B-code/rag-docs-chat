from fastapi.testclient import TestClient

from app.main import app, get_llm, get_vectorstore


class FakeVectorStore:
    def __init__(self):
        self.documents = []

    def ingest(self, chunks):
        self.documents.extend(chunks)
        return len(chunks)

    def search(self, question, top_k=3):
        return [{"text": "Paris is the capital of France.", "source": "geo.txt", "score": 0.91}]


class FakeLLM:
    def complete(self, messages):
        return "Paris is the capital of France."


def make_client():
    app.dependency_overrides[get_vectorstore] = FakeVectorStore
    app.dependency_overrides[get_llm] = FakeLLM
    return TestClient(app)


def test_health():
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_txt_indexes_chunks():
    client = make_client()
    response = client.post(
        "/ingest",
        files=[("files", ("note.txt", b"Hello docs content here.", "text/plain"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] >= 1
    assert "note.txt" in body["files"]


def test_query_returns_answer_and_sources():
    client = make_client()
    response = client.post("/query", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris is the capital of France."
    assert body["sources"][0]["source"] == "geo.txt"


def test_query_validates_question_required():
    client = make_client()
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
