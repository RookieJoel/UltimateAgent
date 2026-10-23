"""Client for the Agents Course Unit 4 scoring API."""
import shutil
from pathlib import Path

import requests

from agent.config import assets_dir

API_URL = "https://agents-course-unit4-scoring.hf.space"
API_TIMEOUT = 60  # seconds
GAIA_DATASET = "gaia-benchmark/GAIA"
GAIA_FILES_PREFIX = "2023/validation"


def fetch_questions() -> list[dict]:
    response = requests.get(f"{API_URL}/questions", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


def download_attachment(task_id: str, file_name: str) -> Path:
    """Download a task's attached file into the assets folder (skipped if already there).

    The course API sometimes answers 404 ("No file path associated with task_id"); then the
    same file is taken from the gated GAIA dataset on the Hub (needs HF_TOKEN with access).
    """
    target_dir = assets_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(file_name).name  # .name drops any directory part
    if target.exists():
        return target
    response = requests.get(f"{API_URL}/files/{task_id}", timeout=API_TIMEOUT)
    if response.status_code == 404:
        return _download_from_hub(target)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def _download_from_hub(target: Path) -> Path:
    from huggingface_hub import hf_hub_download  # reads HF_TOKEN from the environment

    cached = hf_hub_download(GAIA_DATASET, f"{GAIA_FILES_PREFIX}/{target.name}", repo_type="dataset")
    shutil.copyfile(cached, target)
    return target


def submit_answers(username: str, agent_code: str, answers: list[dict]) -> dict:
    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    response = requests.post(f"{API_URL}/submit", json=payload, timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()
