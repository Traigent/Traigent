from traigent.connectors.ports import (
    ArtifactStore,
    DatasetSink,
    DatasetSource,
    DeploymentTarget,
    EventSource,
    ExperimentSink,
    ScoreSink,
    ScoreSource,
    TraceSource,
)


def test_port_protocols_are_runtime_checkable_shapes():
    assert all(
        proto._is_protocol
        for proto in (
            TraceSource,
            ScoreSource,
            ScoreSink,
            DatasetSource,
            DatasetSink,
            ExperimentSink,
            ArtifactStore,
            DeploymentTarget,
            EventSource,
        )
    )
