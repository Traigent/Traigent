from traigent.connectors.models import ConnectionRef
from traigent.connectors.privacy import CustomerSideMinter


def test_connection_minter_is_deterministic_within_connection():
    minter = CustomerSideMinter(ConnectionRef("langfuse"), b"x" * 32)
    assert minter.digest("id") == minter.digest("id")
