# Traceability Schema

Simple YAML/JSON conventions used in `docs/traceability/`.

## Requirements (`requirements.yml`)
- `id` (string, required) — `REQ-XXX`.
- `name` (string, required) — short label for UI.
- `title` (string, required) — full title.
- `description` (string, optional).
- `acceptance_criteria` (list[string], optional).
- `tags` (list[string], optional).
- `priority` (string, optional).
- `type` (string, optional).
- `status` (string, optional).

## Functionalities (`functionalities.yml`)
- `id` (string, required) — `FUNC-XXX`.
- `name` (string, required) — short label for UI.
- `description` (string, optional).
- `requirements` (list[string], optional) — references `REQ-*`.
- `concepts` (list[string], optional) — references `CONC-*`.
- `syncs` (list[string], optional) — references `SYNC-*`.
- `tags` (list[string], optional).
- `layer` (string, optional) — `epic|feature|subfeature|infra`.
- `status` (string, optional).

## Concepts (`concepts/*.yml`)
- `id` (string, required) — `CONC-*`.
- `name` (string, required).
- `purpose` (string, required).
- `type_params` (list[string], optional).
- `state` (list[object], optional) — each with `name`, `type`, `description`.
- `actions` (list[object], optional) — each with `name`, `inputs`, `outputs`, `description`.
- `operational_principle` (list[string], optional).
- `version` (int, optional).
- `status` (string, optional).

## Syncs (`syncs/*.yml`)
- `id` (string, required) — `SYNC-*`.
- `name` (string, required).
- `functionalities` (list[string], optional) — references `FUNC-*`.
- `when`/`where`/`then` (lists, optional) — rule clauses with `concept`, `action`, `args`/`bind`.
- `version` (int, optional).
- `status` (string, optional).

## Code references (generated)
- `docs/traceability/reports/code_summaries.json` — per file/symbol summary; fields: `file`, `doc`, `classes`, `functions`, `concept_id`, `functionalities`, `requirements`, `source`.
- `docs/traceability/reports/trace_links.json` — manual/proposed links: `code_unit`, `concept_id`, `functionalities`, `requirements`, `status`, `source`.

These schemas keep labels available (`name`) for visualization and grouping, while preserving richer text (`title`, `description`) for detail panes and analysis.
