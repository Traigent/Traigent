"""Test error handling and recovery for Optuna integration."""

from __future__ import annotations

import tempfile
import time
from unittest.mock import MagicMock, patch

import optuna
import pytest

from traigent.optimizers.optuna_checkpoint import OptunaCheckpointManager
from traigent.optimizers.optuna_coordinator import (
    BatchOptimizer,
    EdgeExecutor,
    OptunaCoordinator,
    ResilientCoordinator,
)


class TestOptunaErrorRecovery:
    """Test error handling and recovery mechanisms for Optuna optimization."""

    def test_checkpoint_save_and_restore(self):
        """Test checkpoint saving and restoration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = OptunaCheckpointManager(checkpoint_dir=tmpdir)

            # Create a study with some trials
            search_space = {
                "param": optuna.distributions.FloatDistribution(0, 1),
            }

            coordinator = OptunaCoordinator(
                directions=["maximize"], search_space=search_space
            )

            # Run some trials
            trial_results = []
            for _i in range(5):
                configs, _ = coordinator.ask_batch(n_suggestions=1)
                trial_id = configs[0]["_trial_id"]
                value = configs[0]["param"] * 2  # Simple objective
                coordinator.tell_result(trial_id, value)
                trial_results.append((trial_id, value))

            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                study=coordinator.study,
                pending_trials=coordinator.pending_trials,
                metadata={
                    "iteration": 5,
                    "best_value": max(v for _, v in trial_results),
                },
            )

            assert checkpoint_path.exists()

            # Load checkpoint
            loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)

            assert loaded_data["metadata"]["iteration"] == 5
            assert len(loaded_data["completed_trials"]) == 5
            assert loaded_data["metadata"]["best_value"] == max(
                v for _, v in trial_results
            )

    def test_crash_recovery_with_checkpoints(self):
        """Test recovery from crash using checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = OptunaCheckpointManager(checkpoint_dir=tmpdir)

            # Simulate optimization that crashes
            search_space = {
                "x": optuna.distributions.FloatDistribution(-5, 5),
                "crash_trial": optuna.distributions.IntDistribution(0, 10),
            }

            coordinator = OptunaCoordinator(
                directions=["minimize"], search_space=search_space
            )

            completed_before_crash = []

            # Run trials until "crash"
            for _i in range(7):
                configs, _ = coordinator.ask_batch(n_suggestions=1)
                config = configs[0]
                trial_id = config["_trial_id"]

                if config["crash_trial"] == 5:
                    # Simulate crash - save checkpoint first
                    checkpoint_manager.save_checkpoint(
                        study=coordinator.study,
                        pending_trials=coordinator.pending_trials,
                        metadata={"crashed_at_trial": trial_id},
                    )
                    break

                # Complete trial normally
                value = config["x"] ** 2
                coordinator.tell_result(trial_id, value)
                completed_before_crash.append(trial_id)

            # Simulate recovery - create new coordinator and load checkpoint
            latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
            assert latest_checkpoint is not None

            loaded_data = checkpoint_manager.load_checkpoint(latest_checkpoint)

            # Create new coordinator with recovered state
            new_coordinator = OptunaCoordinator(
                directions=["minimize"], search_space=search_space
            )

            # Restore completed trials
            for trial_data in loaded_data["completed_trials"]:
                # Recreate trials in new study
                new_coordinator.study.add_trial(
                    optuna.trial.create_trial(
                        params=trial_data["params"],
                        distributions=search_space,
                        values=trial_data["values"],
                        state=optuna.trial.TrialState.COMPLETE,
                    )
                )

            # Continue optimization
            for _i in range(3):
                configs, _ = new_coordinator.ask_batch(n_suggestions=1)
                trial_id = configs[0]["_trial_id"]
                value = configs[0]["x"] ** 2
                new_coordinator.tell_result(trial_id, value)

            # Verify we have all trials
            assert len(new_coordinator.study.trials) >= len(completed_before_crash) + 3

    def test_edge_executor_offline_queue(self):
        """Test EdgeExecutor handling offline scenarios."""
        coordinator_url = "http://fake-coordinator.local"
        device_id = "edge-device-001"

        executor = EdgeExecutor(coordinator_url, device_id)
        executor.n_steps = 3

        # Mock offline state
        executor.is_online = MagicMock(return_value=False)

        # Execute trial while offline
        config = {"param": 0.5, "_trial_id": 42}

        # Mock execution steps
        def mock_run_step(cfg, step):
            return 0.5 + step * 0.1  # Increasing values

        executor.run_step = mock_run_step
        executor.compute_final = lambda cfg: 0.9

        # This should queue results locally
        with patch.object(executor, "report_final") as mock_report:
            mock_report.return_value = None
            executor.execute_trial(config)

            # Verify results were queued
            assert len(executor.local_queue) == 3  # One per step
            assert all(item["trial_id"] == 42 for item in executor.local_queue)

        # Now simulate coming back online
        executor.is_online = MagicMock(return_value=True)

        with patch.object(executor, "report_to_coordinator") as mock_sync:
            executor.sync_offline_results()

            # Verify all queued results were synced
            assert mock_sync.call_count == 3
            assert len(executor.local_queue) == 0

    def test_resilient_coordinator_retry_logic(self):
        """Test ResilientCoordinator retry mechanism."""
        coordinator = ResilientCoordinator(
            retry_policy={"max_retries": 3, "backoff_base": 0.01, "max_backoff": 0.1}
        )

        # Mock network behavior - fail twice, then succeed
        attempt_count = 0

        def mock_send_result(trial_id, value):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network unavailable")
            return True

        coordinator.send_result = mock_send_result
        coordinator.is_online = MagicMock(return_value=True)
        coordinator.flush_offline_queue = MagicMock()

        # Should retry and eventually succeed
        success = coordinator.report_with_retry(trial_id=1, value=0.95)

        assert success is True
        assert attempt_count == 3
        coordinator.flush_offline_queue.assert_called_once()

    def test_resilient_coordinator_offline_queue(self):
        """Test offline queuing when all retries fail."""
        coordinator = ResilientCoordinator(
            retry_policy={"max_retries": 2, "backoff_base": 0.01, "max_backoff": 0.1}
        )

        # Always fail
        coordinator.send_result = MagicMock(side_effect=ConnectionError("Network down"))
        coordinator.is_online = MagicMock(return_value=False)

        # Should queue after retries fail
        success = coordinator.report_with_retry(trial_id=5, value=0.75)

        assert success is False
        assert len(coordinator.offline_queue) == 1
        assert coordinator.offline_queue[0]["trial_id"] == 5
        assert coordinator.offline_queue[0]["value"] == 0.75

    def test_batch_optimizer_timeout_handling(self):
        """Test BatchOptimizer handling worker timeouts."""

        def slow_worker(config):
            time.sleep(2)  # Will timeout
            return {"value": config["param"] * 2}

        search_space = {"param": optuna.distributions.FloatDistribution(0, 1)}

        batch_optimizer = BatchOptimizer(
            config_space=search_space,
            objectives=["value"],
            n_workers=2,
            worker_fn=slow_worker,
            trial_timeout=0.1,  # Very short timeout
        )

        # Mock coordinator to track failures
        batch_optimizer.coordinator = MagicMock()
        batch_optimizer.coordinator.ask_batch = MagicMock(
            return_value=(
                [{"param": 0.5, "_trial_id": 1}, {"param": 0.7, "_trial_id": 2}],
                [MagicMock(), MagicMock()],
            )
        )

        with patch.object(batch_optimizer, "dispatch_to_worker") as mock_dispatch:
            # Create mock futures that timeout
            mock_future = MagicMock()
            mock_future.get = MagicMock(side_effect=TimeoutError())
            mock_dispatch.return_value = mock_future

            batch_optimizer.optimize_batch(n_trials=2)

            # Verify timeouts were handled
            assert batch_optimizer.coordinator.tell_failure.call_count == 2
            for call in batch_optimizer.coordinator.tell_failure.call_args_list:
                assert "timed out" in call[0][1].lower()

    def test_worker_exception_handling(self):
        """Test handling of worker exceptions."""

        def faulty_worker(config):
            if config["param"] > 0.5:
                raise RuntimeError(f"Worker failed with param={config['param']}")
            return {"value": config["param"] * 2}

        search_space = {"param": optuna.distributions.FloatDistribution(0, 1)}

        batch_optimizer = BatchOptimizer(
            config_space=search_space,
            objectives=["value"],
            n_workers=1,
            worker_fn=faulty_worker,
        )

        # Run optimization with some failing trials
        with patch.object(batch_optimizer.coordinator, "tell_failure") as mock_fail:
            # Mock to control parameter values
            with patch.object(batch_optimizer.coordinator, "ask_batch") as mock_ask:
                mock_ask.side_effect = [
                    ([{"param": 0.3, "_trial_id": 1}], [MagicMock()]),  # Success
                    ([{"param": 0.7, "_trial_id": 2}], [MagicMock()]),  # Fail
                    ([{"param": 0.2, "_trial_id": 3}], [MagicMock()]),  # Success
                ]

                batch_optimizer.optimize_batch(n_trials=3)

                # One trial should have failed
                assert mock_fail.call_count == 1
                assert "Worker failed" in mock_fail.call_args[0][1]

    def test_checkpoint_recovery_with_pruned_trials(self):
        """Test checkpoint recovery including pruned trials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = OptunaCheckpointManager(checkpoint_dir=tmpdir)

            search_space = {"param": optuna.distributions.FloatDistribution(0, 1)}

            coordinator = OptunaCoordinator(
                directions=["maximize"], search_space=search_space
            )

            # Create mix of completed and pruned trials
            trial_states = []

            for i in range(6):
                configs, _ = coordinator.ask_batch(n_suggestions=1)
                trial_id = configs[0]["_trial_id"]

                if i % 3 == 0:
                    # Prune this trial
                    coordinator.tell_result(
                        trial_id,
                        None,
                        metadata={"state": "pruned", "pruned_at_step": 5},
                    )
                    trial_states.append(("pruned", trial_id))
                else:
                    # Complete normally
                    coordinator.tell_result(trial_id, configs[0]["param"])
                    trial_states.append(("completed", trial_id))

            # Save checkpoint
            checkpoint_path = checkpoint_manager.save_checkpoint(
                study=coordinator.study,
                pending_trials={},
                metadata={"trial_states": trial_states},
            )

            # Load and verify
            loaded_data = checkpoint_manager.load_checkpoint(checkpoint_path)

            completed = loaded_data["completed_trials"]
            pruned = loaded_data["pruned_trials"]

            assert len(completed) == 4  # 6 total - 2 pruned
            assert len(pruned) == 2

    def test_storage_failure_recovery(self):
        """Test recovery from storage backend failures."""
        # Use SQLite storage that we can corrupt
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            storage_url = f"sqlite:///{tmp.name}"

            coordinator = OptunaCoordinator(
                directions=["minimize"],
                search_space={"param": optuna.distributions.FloatDistribution(0, 1)},
                storage=storage_url,
            )

            # Run some trials
            for _ in range(3):
                configs, _ = coordinator.ask_batch(n_suggestions=1)
                coordinator.tell_result(configs[0]["_trial_id"], 0.5)

            # Simulate storage failure by mocking study operations
            with patch.object(
                coordinator.study, "tell", side_effect=Exception("Storage error")
            ):
                configs, _ = coordinator.ask_batch(n_suggestions=1)

                # This should handle the storage error gracefully
                with pytest.raises(Exception, match="Storage error"):
                    coordinator.tell_result(configs[0]["_trial_id"], 0.5)

            # Verify we can recover and continue
            configs, _ = coordinator.ask_batch(n_suggestions=1)
            coordinator.tell_result(configs[0]["_trial_id"], 0.7)

            # Should have trials from before and after the failure
            assert len(coordinator.study.trials) >= 4

    def test_partial_batch_failure(self):
        """Test handling when some workers in a batch fail."""

        def selective_worker(config):
            # Fail for specific parameter values
            if 0.4 < config["param"] < 0.6:
                raise ValueError("Parameter in failure zone")
            return {"value": config["param"] ** 2}

        search_space = {"param": optuna.distributions.FloatDistribution(0, 1)}

        batch_optimizer = BatchOptimizer(
            config_space=search_space,
            objectives=["value"],
            n_workers=3,
            worker_fn=selective_worker,
        )

        # Track successes and failures
        success_count = 0
        failure_count = 0

        def mock_tell_result(trial_id, value, *args):
            nonlocal success_count
            success_count += 1

        def mock_tell_failure(trial_id, error):
            nonlocal failure_count
            failure_count += 1

        batch_optimizer.coordinator.tell_result = mock_tell_result
        batch_optimizer.coordinator.tell_failure = mock_tell_failure

        # Run batch with mixed results
        batch_optimizer.optimize_batch(n_trials=9)

        # Should have both successes and failures
        assert success_count > 0
        assert failure_count > 0
        assert success_count + failure_count == 9

    def test_checkpoint_manager_auto_cleanup(self):
        """Test automatic cleanup of old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = OptunaCheckpointManager(
                checkpoint_dir=tmpdir, max_checkpoints=3
            )

            # Create more checkpoints than the limit
            for i in range(5):
                checkpoint_manager.save_checkpoint(
                    study=MagicMock(),
                    pending_trials={},
                    metadata={"iteration": i},
                )
                time.sleep(0.01)  # Ensure different timestamps

            # Should only keep the latest 3
            checkpoints = checkpoint_manager.list_checkpoints()
            assert len(checkpoints) == 3

            # Verify these are the latest ones
            loaded = checkpoint_manager.load_checkpoint(checkpoints[-1])
            assert loaded["metadata"]["iteration"] == 4  # Latest

    def test_distributed_failure_detection(self):
        """Test detection of failures in distributed setup."""

        class DistributedCoordinator(OptunaCoordinator):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.worker_heartbeats = {}
                self.heartbeat_timeout = 5.0

            def register_worker(self, worker_id):
                self.worker_heartbeats[worker_id] = time.time()

            def heartbeat(self, worker_id):
                self.worker_heartbeats[worker_id] = time.time()

            def detect_failed_workers(self):
                current_time = time.time()
                failed = []
                for worker_id, last_heartbeat in self.worker_heartbeats.items():
                    if current_time - last_heartbeat > self.heartbeat_timeout:
                        failed.append(worker_id)
                return failed

        coordinator = DistributedCoordinator(
            directions=["maximize"],
            search_space={"param": optuna.distributions.FloatDistribution(0, 1)},
        )

        # Register workers
        coordinator.register_worker("worker-1")
        coordinator.register_worker("worker-2")

        # Simulate heartbeats
        coordinator.heartbeat("worker-1")
        time.sleep(0.1)

        # Worker-2 stops sending heartbeats
        coordinator.worker_heartbeats["worker-2"] = time.time() - 10  # Old timestamp

        # Detect failures
        failed_workers = coordinator.detect_failed_workers()

        assert "worker-2" in failed_workers
        assert "worker-1" not in failed_workers
