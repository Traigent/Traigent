from traigent.integrations.observability.workflow_traces import (
    _redact_observability_object,
)

print(_redact_observability_object({"api_key": {"nested": "secret_value"}}))
