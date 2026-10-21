# One table across a run group

A run group collects the runs of one agent and dataset. `experiment_groups.table`
reads every configuration-run in the group and returns a single table, so you can
compare runs without stitching reports together. It is read-only and uses the same
credentials as the other analytics reads.

```python
table = await client.experiment_groups.table("grp_123", "proj_abc")
for row in table:
    print(row.run_id, row.run_name, row.config, row.metrics, row.cost)

df = table.to_dataframe()  # needs: pip install "traigent[analytics]"
```

Each `RunGroupRow` has `run_id`, `run_name`, `configuration_run_id`, `experiment_id`,
`config` (dict), `metrics` (dict of numeric measures), `cost`, `cost_unmeasured`,
`cost_floor`, `status` and `created_at`.

- **Cost is never invented.** If a run's cost was not measured (or not priced),
  `cost` is `None` and `cost_unmeasured` is `True`. `cost_floor` stays `None` until the
  backend reports a known-spend floor.
- **Rows are source runs.** Nothing is merged or deduplicated by configuration.
- **Limits.** Rows are fetched `page_size` (default 100) at a time, up to `max_rows`
  (default 10,000). `table.truncated` is `True` when the limit stopped the fetch.
- **Errors.** A backend refusal (for example 401/403) raises `AnalyticsClientError`
  that carries only the HTTP status, never request or response content.
