from __future__ import annotations

import pytest

from agent_harness import FakeProvider, Harness


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def harness(provider: FakeProvider) -> Harness:
    return Harness.testing(provider)
