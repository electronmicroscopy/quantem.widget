from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_pr_cleanup_guidance_is_safe_and_agent_readable() -> None:
    workflow = (REPO / ".github/workflows/pr-cleanup-guidance.yml").read_text()
    ci_workflow = (REPO / ".github/workflows/widget-ci.yml").read_text()

    assert "pull_request_target:" in workflow
    assert "types: [opened, reopened, closed]" in workflow
    assert "issues: write" in workflow
    assert "actions/checkout" not in workflow
    assert "quantem-pr-cleanup-guidance:v1" in workflow
    assert "Agent-readable" not in workflow
    assert "JSON.stringify" not in workflow
    assert "### Repository cleanup" not in workflow
    assert "tasks.map((task) => `- ${task}`)" in workflow
    assert "pr.merged" in workflow
    assert 'action === "closed" && !pr.merged' in workflow
    assert "issues.deleteComment" in workflow
    assert ci_workflow.count('".github/PULL_REQUEST_TEMPLATE.md"') == 2
    assert ci_workflow.count('".github/workflows/pr-cleanup-guidance.yml"') == 2


def test_pr_template_requires_private_information_preflight() -> None:
    template = (REPO / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    normalized = " ".join(template.split())

    for required in (
        "personal names or usernames",
        "private email addresses",
        "absolute machine paths",
        "hostnames",
        "credentials or tokens",
        "private sample or dataset identifiers",
    ):
        assert required in normalized
