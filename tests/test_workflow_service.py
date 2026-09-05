from types import SimpleNamespace

import pytest

from src.services.workflow import (
    extract_text,
    get_interrupt_payload,
    make_config,
    prepare_repository_runtime,
    resume_workflow,
)


def test_extract_text_handles_string_and_text_blocks():
    assert extract_text("hello") == "hello"
    assert extract_text([{"text": "hello"}, {"text": "world"}]) == "hello\nworld"


def test_make_config_uses_supplied_thread_id():
    config = make_config(thread_id="repo-session")

    assert config["configurable"]["thread_id"] == "repo-session"
    assert config["recursion_limit"] == 30


def test_get_interrupt_payload_returns_first_interrupt_value():
    result = {
        "__interrupt__": [SimpleNamespace(value={"type": "human_approval"})]
    }

    assert get_interrupt_payload(result) == {"type": "human_approval"}
    assert get_interrupt_payload({}) is None


def test_prepare_repository_runtime_rejects_repository_without_supported_files(tmp_path):
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    (empty_repo / "image.bin").write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="No supported source files"):
        prepare_repository_runtime(str(empty_repo), source_type="local")


def test_resume_workflow_requires_feedback_for_rejection():
    class FakeGraph:
        def invoke(self, *_args, **_kwargs):
            raise AssertionError("Graph should not run without rejection feedback")

    with pytest.raises(ValueError, match="Add feedback"):
        resume_workflow(
            FakeGraph(),
            {"configurable": {"thread_id": "test"}},
            approved=False,
            feedback="",
        )


def test_resume_workflow_uses_default_feedback_for_approval():
    class FakeGraph:
        def __init__(self):
            self.command = None
            self.config = None

        def invoke(self, command, config=None):
            self.command = command
            self.config = config
            return {"human_approved": True}

    graph = FakeGraph()
    config = {"configurable": {"thread_id": "test"}}

    result = resume_workflow(graph, config, approved=True)

    assert result["human_approved"] is True
    assert graph.config == config
