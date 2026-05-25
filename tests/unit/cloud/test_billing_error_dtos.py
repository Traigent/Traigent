from traigent.cloud.dtos import (
    QuotaExceededErrorDTO,
    WalletInsufficientBalanceErrorDTO,
    WalletTopUpPackDTO,
)


def test_quota_exceeded_error_dto_matches_schema_shape():
    dto = QuotaExceededErrorDTO(
        resource_type="api_calls",
        current_usage=25,
        limit=25,
        reset_at="2026-06-01T00:00:00+00:00",
        message="API calls limit reached.",
    )

    assert dto.to_dict() == {
        "error_code": "quota_exceeded",
        "resource_type": "api_calls",
        "current_usage": 25,
        "limit": 25,
        "reset_at": "2026-06-01T00:00:00+00:00",
        "upgrade_url": "/billing",
        "message": "API calls limit reached.",
    }


def test_wallet_insufficient_balance_error_dto_matches_schema_shape():
    dto = WalletInsufficientBalanceErrorDTO(
        available_usd="0.00",
        required_usd="0.05",
        operation_id="op_123",
        operation_group_id="group_123",
        message="Add wallet credits to continue generation.",
    )

    assert dto.to_dict()["error_code"] == "wallet_insufficient_balance"
    assert dto.to_dict()["available_usd"] == "0.00"
    assert dto.to_dict()["required_usd"] == "0.05"


def test_wallet_top_up_pack_dto_exposes_no_price_id():
    dto = WalletTopUpPackDTO(pack_id="starter", credit_usd="5.00")

    assert dto.to_dict() == {"pack_id": "starter", "credit_usd": "5.00"}
