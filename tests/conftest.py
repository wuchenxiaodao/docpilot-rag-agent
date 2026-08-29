import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# tests/ 不是包，pytest 只把 src 加进 sys.path（pyproject pythonpath=["src"]），
# 而部分用例顶层导入 experiments.* / scripts.* —— 这里把项目根补进搜索路径。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_addoption(parser):
    parser.addoption(
        "--run-docker", action="store_true", default=False, help="run docker integration tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker containers")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-docker"):
        skip_docker = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)


@pytest.fixture
def mock_env():
    """Fixture to ensure environment is clean for each test."""
    # 全清环境变量会让 streamlit 脚本线程在 Path.home() 解析 ~/.streamlit
    # 时崩溃（Could not determine home directory），AppTest 等不到完成事件
    # 只能超时。保留 home 目录与 Windows 系统基本键，其余照清。
    keep_keys = ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "SYSTEMROOT")
    keep = {k: os.environ[k] for k in keep_keys if k in os.environ}
    with patch.dict(os.environ, keep, clear=True):
        yield
