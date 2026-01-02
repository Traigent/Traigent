# Traigent Architecture Analysis Report

Generated: 2025-11-29T22:52:19.542713

## Summary

| Metric | Value |
|--------|-------|
| Total Python Files | 223 |
| Total Lines of Code | 97,224 |
| Total Classes | 624 |
| Total Functions | 3300 |
| Internal Dependencies | 527 |

## Package Structure

```
└── 📦 traigent/ [223f, 97,224L, 624C, 3300F]
    ├── 📋 __init__.py [136L, 0C, 0F] 📝
    ├── 📄 _version.py [54L, 0C, 2F] 📝
    ├── 🏗️ optigen_integration.py [599L, 1C, 11F] 📝
    ├── 📦 agents/ [5f, 2,679L, 14C, 93F]
    │   ├── 📋 __init__.py [65L, 0C, 0F] 📝
    │   ├── 🏗️ config_mapper.py [647L, 3C, 19F] 📝
    │   ├── 🏗️ executor.py [296L, 2C, 14F] 📝
    │   ├── 🏗️ platforms.py [815L, 3C, 29F] 📝
    │   └── 🏗️ specification_generator.py [856L, 6C, 31F] 📝
    ├── 📦 analytics/ [7f, 7,153L, 45C, 210F]
    │   ├── 📋 __init__.py [95L, 0C, 0F] 📝
    │   ├── 🏗️ anomaly.py [1115L, 11C, 40F] 📝
    │   ├── 🏗️ cost_optimization.py [1168L, 9C, 28F] 📝
    │   ├── 🏗️ intelligence.py [1861L, 3C, 62F] 📝
    │   ├── 🏗️ meta_learning.py [794L, 8C, 27F] 📝
    │   ├── 🏗️ predictive.py [1520L, 8C, 41F] 📝
    │   └── 🏗️ scheduling.py [600L, 6C, 12F] 📝
    ├── 📦 api/ [6f, 3,201L, 21C, 89F]
    │   ├── 📋 __init__.py [38L, 0C, 0F] 📝
    │   ├── 🏗️ config_builder.py [231L, 1C, 13F] 📝
    │   ├── 🏗️ decorators.py [962L, 5C, 8F] 📝
    │   ├── 📄 functions.py [605L, 0C, 17F] 📝
    │   ├── 🏗️ parameter_validator.py [298L, 2C, 11F] 📝
    │   └── 🏗️ types.py [1067L, 13C, 40F] 📝
    ├── 📦 cli/ [7f, 3,240L, 4C, 67F]
    │   ├── 📋 __init__.py [15L, 0C, 0F] 📝
    │   ├── 🏗️ auth_commands.py [679L, 1C, 17F] 📝
    │   ├── 📄 function_discovery.py [277L, 0C, 5F] 📝
    │   ├── 📄 local_commands.py [641L, 0C, 11F] 📝
    │   ├── 📄 main.py [1078L, 0C, 16F] 📝
    │   ├── 🏗️ optimization_validator.py [426L, 1C, 10F] 📝
    │   └── 🏗️ validation_types.py [124L, 2C, 8F] 📝
    ├── 📦 cloud/ [29f, 19,110L, 128C, 647F]
    │   ├── 📋 __init__.py [32L, 0C, 0F] 📝
    │   ├── 🏗️ _aiohttp_compat.py [63L, 5C, 5F] 📝
    │   ├── 🏗️ api_operations.py [765L, 7C, 27F] 📝
    │   ├── 🏗️ auth.py [2208L, 12C, 91F] 📝
    │   ├── 🏗️ backend_bridges.py [804L, 5C, 25F] 📝
    │   ├── 🏗️ backend_client.py [921L, 2C, 59F] 📝
    │   ├── 🏗️ backend_components.py [242L, 4C, 11F] 📝
    │   ├── 🏗️ backend_synchronizer.py [701L, 3C, 15F] 📝
    │   ├── 🏗️ billing.py [1283L, 11C, 39F] 📝
    │   ├── 🏗️ client.py [1671L, 5C, 55F] 📝
    │   ├── 🏗️ cloud_operations.py [132L, 1C, 3F] 📝
    │   ├── 🏗️ credential_manager.py [214L, 1C, 8F] 📝
    │   ├── 🏗️ dataset_converter.py [796L, 5C, 26F] 📝
    │   ├── 🏗️ dtos.py [338L, 5C, 10F] 📝
    │   ├── 🏗️ event_manager.py [549L, 4C, 19F] 📝
    │   ├── 🏗️ integration_manager.py [774L, 4C, 22F] 📝
    │   ├── 🏗️ models.py [364L, 20C, 6F] 📝
    │   ├── 🏗️ optimizer_client.py [396L, 1C, 14F] 📝
    │   ├── 🏗️ privacy_operations.py [321L, 1C, 4F] 📝
    │   ├── 🏗️ production_mcp_client.py [880L, 6C, 31F] 📝
    │   ├── 🏗️ resilient_client.py [454L, 4C, 11F] 📝
    │   ├── 🏗️ service.py [429L, 3C, 8F] 📝
    │   ├── 🏗️ session_operations.py [831L, 1C, 21F] 📝
    │   ├── 🏗️ sessions.py [1227L, 6C, 60F] 📝
    │   ├── 🏗️ subset_selection.py [596L, 8C, 20F] 📝
    │   ├── 🏗️ sync_manager.py [589L, 1C, 15F] 📝
    │   ├── 🏗️ trial_operations.py [845L, 1C, 20F] 📝
    │   ├── 🏗️ trial_tracker.py [491L, 2C, 18F] 📝
    │   └── 📄 validators.py [194L, 0C, 4F] 📝
    ├── 📦 config/ [12f, 3,418L, 25C, 134F]
    │   ├── 📋 __init__.py [42L, 0C, 0F] 📝
    │   ├── 🏗️ api_keys.py [87L, 1C, 5F] 📝
    │   ├── 🏗️ ast_transformer.py [518L, 2C, 11F] 📝
    │   ├── 🏗️ backend_config.py [253L, 1C, 14F] 📝
    │   ├── 🏗️ context.py [236L, 3C, 18F] 📝
    │   ├── 🏗️ feature_flags.py [201L, 3C, 13F] 📝
    │   ├── 🏗️ parallel.py [315L, 2C, 11F] 📝
    │   ├── 🏗️ providers.py [911L, 5C, 26F] 📝
    │   ├── 📄 runtime_injector.py [96L, 0C, 2F] 📝
    │   ├── 🏗️ seamless_injection.py [141L, 4C, 10F] 📝
    │   ├── 🏗️ seamless_optuna_adapter.py [105L, 1C, 4F] 📝
    │   └── 🏗️ types.py [513L, 3C, 20F] 📝
    ├── 📦 core/ [34f, 11,897L, 72C, 411F]
    │   ├── 📋 __init__.py [12L, 0C, 0F] 📝
    │   ├── 🏗️ backend_session_manager.py [642L, 1C, 9F] 📝
    │   ├── 🏗️ cache_policy.py [155L, 1C, 7F] 📝
    │   ├── 🏗️ config_builder.py [569L, 2C, 21F] 📝
    │   ├── 📄 constants.py [258L, 0C, 3F] 📝
    │   ├── 🏗️ evaluator_wrapper.py [485L, 1C, 10F] 📝
    │   ├── 🏗️ llm_processor.py [337L, 1C, 14F] 📝
    │   ├── 🏗️ logger_facade.py [131L, 1C, 10F] 📝
    │   ├── 🏗️ mandatory_metrics.py [134L, 2C, 10F] 📝
    │   ├── 📄 metadata_helpers.py [302L, 0C, 4F] 📝
    │   ├── 🏗️ metric_registry.py [78L, 2C, 11F] 📝
    │   ├── 🏗️ metrics_aggregator.py [115L, 1C, 2F] 📝
    │   ├── 🏗️ objectives.py [703L, 3C, 24F] 📝
    │   ├── 🏗️ optimized_function.py [1672L, 1C, 50F] 📝
    │   ├── 🏗️ orchestrator.py [1831L, 2C, 59F] 📝
    │   ├── 📄 orchestrator_helpers.py [351L, 0C, 9F] 📝
    │   ├── 🏗️ parallel_execution_manager.py [273L, 3C, 9F] 📝
    │   ├── 🏗️ progress_manager.py [259L, 2C, 16F] 📝
    │   ├── 🏗️ pruning_progress_tracker.py [280L, 1C, 10F] 📝
    │   ├── 🏗️ refactoring_utils.py [248L, 1C, 9F] 📝
    │   ├── 🏗️ result_selection.py [127L, 1C, 2F] 📝
    │   ├── 🏗️ sample_budget.py [247L, 4C, 19F] 📝
    │   ├── 🏗️ session_context.py [26L, 1C, 0F] 📝
    │   ├── 🏗️ stop_condition_manager.py [136L, 1C, 7F] 📝
    │   ├── 🏗️ stop_conditions.py [261L, 5C, 16F] 📝
    │   ├── 🏗️ trial_context.py [31L, 1C, 0F] 📝
    │   ├── 📄 trial_result_factory.py [206L, 0C, 3F] 📝
    │   ├── 🏗️ types.py [319L, 3C, 15F] 📝
    │   ├── 🏗️ types_ext.py [471L, 27C, 9F] 📝
    │   ├── 📄 utils.py [665L, 0C, 22F] 📝
    │   └── 📦 samplers/ [4f, 573L, 4C, 31F]
    │       ├── 📋 __init__.py [36L, 0C, 0F] 📝
    │       ├── 🏗️ base.py [73L, 1C, 10F] 📝
    │       ├── 🏗️ factory.py [40L, 1C, 2F] 📝
    │       └── 🏗️ random_sampler.py [424L, 2C, 19F] 📝
    ├── 📦 evaluators/ [6f, 4,766L, 25C, 138F]
    │   ├── 📋 __init__.py [13L, 0C, 0F] 📝
    │   ├── 🏗️ base.py [2068L, 6C, 59F] 📝
    │   ├── 🏗️ dataset_registry.py [168L, 1C, 4F] 📝
    │   ├── 🏗️ local.py [1095L, 2C, 24F] 📝
    │   ├── 🏗️ metrics.py [234L, 2C, 6F] 📝
    │   └── 🏗️ metrics_tracker.py [1188L, 14C, 45F] 📝
    ├── 📦 experimental/ [1f, 33L, 0C, 0F]
    │   └── 📋 __init__.py [33L, 0C, 0F] 📝
    ├── 📦 integrations/ [26f, 8,241L, 35C, 286F]
    │   ├── 📋 __init__.py [212L, 0C, 2F] 📝
    │   ├── 🏗️ azure_openai_client.py [84L, 2C, 3F] 📝
    │   ├── 🏗️ base.py [313L, 1C, 23F] 📝
    │   ├── 🏗️ base_plugin.py [441L, 4C, 17F] 📝
    │   ├── 🏗️ bedrock_client.py [338L, 2C, 10F] 📝
    │   ├── 🏗️ config.py [388L, 4C, 9F] 📝
    │   ├── 🏗️ framework_override.py [1423L, 2C, 52F] 📝
    │   ├── 🏗️ google_gemini_client.py [90L, 2C, 3F] 📝
    │   ├── 🏗️ plugin_registry.py [391L, 1C, 17F] 📝
    │   ├── 📦 llms/ [9f, 2,035L, 7C, 66F]
    │   │   ├── 📋 __init__.py [4L, 0C, 0F] 📝
    │   │   ├── 🏗️ anthropic_plugin.py [198L, 1C, 6F] 📝
    │   │   ├── 🏗️ langchain_plugin.py [393L, 1C, 8F] 📝
    │   │   ├── 🏗️ llamaindex_plugin.py [351L, 1C, 8F] 📝
    │   │   ├── 🏗️ openai.py [303L, 1C, 13F] 📝
    │   │   ├── 🏗️ openai_plugin.py [203L, 1C, 7F] 📝
    │   │   └── 📦 langchain/ [3f, 583L, 2C, 24F]
    │   │       ├── 📋 __init__.py [4L, 0C, 0F] 📝
    │   │       ├── 🏗️ base.py [269L, 1C, 13F] 📝
    │   │       └── 🏗️ discovery.py [310L, 1C, 11F] 📝
    │   ├── 📦 observability/ [3f, 1,242L, 6C, 45F]
    │   │   ├── 📋 __init__.py [4L, 0C, 0F] 📝
    │   │   ├── 🏗️ mlflow.py [558L, 3C, 24F] 📝
    │   │   └── 🏗️ wandb.py [680L, 3C, 21F] 📝
    │   ├── 📦 utils/ [4f, 1,280L, 4C, 39F]
    │   │   ├── 📋 __init__.py [4L, 0C, 0F] 📝
    │   │   ├── 🏗️ discovery.py [703L, 1C, 19F] 📝
    │   │   ├── 🏗️ validation.py [314L, 1C, 11F] 📝
    │   │   └── 🏗️ version_compat.py [259L, 2C, 9F] 📝
    │   └── 📦 vector_stores/ [1f, 4L, 0C, 0F]
    │       └── 📋 __init__.py [4L, 0C, 0F] 📝
    ├── 📦 invokers/ [4f, 826L, 4C, 36F]
    │   ├── 📋 __init__.py [19L, 0C, 0F] 📝
    │   ├── 🏗️ base.py [278L, 2C, 16F] 📝
    │   ├── 🏗️ batch.py [278L, 1C, 10F] 📝
    │   └── 🏗️ local.py [251L, 1C, 10F] 📝
    ├── 📦 metrics/ [3f, 563L, 2C, 19F]
    │   ├── 📋 __init__.py [34L, 0C, 0F] 📝
    │   ├── 🏗️ ragas_metrics.py [473L, 2C, 14F] 📝
    │   └── 📄 registry.py [56L, 0C, 5F] 📝
    ├── 📦 optimizers/ [20f, 7,740L, 46C, 286F]
    │   ├── 📋 __init__.py [50L, 0C, 0F] 📝
    │   ├── 🏗️ base.py [260L, 1C, 13F] 📝
    │   ├── 🏗️ batch_optimizers.py [790L, 4C, 22F] 📝
    │   ├── 🏗️ bayesian.py [562L, 1C, 10F] 📝
    │   ├── 🏗️ benchmarking.py [136L, 1C, 5F] 📝
    │   ├── 🏗️ cloud_optimizer.py [565L, 1C, 15F] 📝
    │   ├── 🏗️ grid.py [266L, 1C, 13F] 📝
    │   ├── 🏗️ interactive_optimizer.py [407L, 2C, 13F] 📝
    │   ├── 🏗️ optuna_adapter.py [156L, 1C, 5F] 📝
    │   ├── 🏗️ optuna_checkpoint.py [331L, 1C, 14F] 📝
    │   ├── 🏗️ optuna_coordinator.py [570L, 6C, 36F] 📝
    │   ├── 🏗️ optuna_optimizer.py [617L, 7C, 19F] 📝
    │   ├── 📄 optuna_utils.py [314L, 0C, 7F] 📝
    │   ├── 🏗️ pruners.py [109L, 2C, 3F] 📝
    │   ├── 🏗️ random.py [175L, 1C, 8F] 📝
    │   ├── 📄 registry.py [183L, 0C, 7F] 📝
    │   ├── 🏗️ remote.py [176L, 2C, 9F] 📝
    │   ├── 🏗️ remote_services.py [1135L, 10C, 39F] 📝
    │   ├── 🏗️ results.py [50L, 2C, 4F] 📝
    │   └── 🏗️ service_registry.py [888L, 3C, 44F] 📝
    ├── 📦 plugins/ [2f, 625L, 6C, 41F]
    │   ├── 📋 __init__.py [40L, 0C, 0F] 📝
    │   └── 🏗️ registry.py [585L, 6C, 41F] 📝
    ├── 📦 security/ [22f, 9,367L, 88C, 333F]
    │   ├── 📋 __init__.py [52L, 0C, 0F] 📝
    │   ├── 🏗️ audit.py [945L, 10C, 47F] 📝
    │   ├── 🏗️ config.py [101L, 2C, 4F] 📝
    │   ├── 🏗️ credentials.py [950L, 5C, 31F] 📝
    │   ├── 🏗️ crypto_utils.py [514L, 6C, 16F] 📝
    │   ├── 🏗️ deployment.py [335L, 8C, 15F] 📝
    │   ├── 🏗️ encryption.py [1260L, 14C, 47F] 📝
    │   ├── 🏗️ enterprise.py [833L, 8C, 26F] 📝
    │   ├── 🏗️ headers.py [309L, 2C, 10F] 📝
    │   ├── 🏗️ input_validation.py [482L, 2C, 15F] 📝
    │   ├── 🏗️ jwt_validator.py [668L, 8C, 16F] 📝
    │   ├── 🏗️ rate_limiter.py [613L, 6C, 20F] 📝
    │   ├── 🏗️ session_manager.py [498L, 2C, 14F] 📝
    │   ├── 🏗️ tenant.py [853L, 9C, 44F] 📝
    │   └── 📦 auth/ [8f, 954L, 6C, 28F]
    │       ├── 📋 __init__.py [50L, 0C, 0F] 📝
    │       ├── 📄 helpers.py [81L, 0C, 3F] 📝
    │       ├── 🏗️ mfa.py [76L, 1C, 4F] 📝
    │       ├── 🏗️ models.py [42L, 1C, 1F] 📝
    │       ├── 🏗️ oidc.py [211L, 1C, 5F] 📝
    │       ├── 🏗️ saml.py [162L, 1C, 6F] 📝
    │       ├── 🏗️ sms.py [191L, 1C, 4F] 📝
    │       └── 🏗️ totp.py [141L, 1C, 5F] 📝
    ├── 📦 storage/ [2f, 748L, 3C, 27F]
    │   ├── 📋 __init__.py [9L, 0C, 0F] 📝
    │   └── 🏗️ local_storage.py [739L, 3C, 27F] 📝
    ├── 📦 telemetry/ [2f, 82L, 1C, 5F]
    │   ├── 📋 __init__.py [6L, 0C, 0F] 📝
    │   └── 🏗️ optuna_metrics.py [76L, 1C, 5F] 📝
    ├── 📦 tvl/ [3f, 703L, 5C, 26F]
    │   ├── 📋 __init__.py [20L, 0C, 0F] 📝
    │   ├── 🏗️ options.py [46L, 1C, 3F] 📝
    │   └── 🏗️ spec_loader.py [637L, 4C, 23F] 📝
    ├── 📦 utils/ [27f, 11,409L, 98C, 425F]
    │   ├── 📋 __init__.py [117L, 0C, 0F] 📝
    │   ├── 🏗️ batch_optimizer_utils.py [347L, 2C, 10F] 📝
    │   ├── 🏗️ batch_processing.py [563L, 7C, 24F] 📝
    │   ├── 🏗️ callbacks.py [647L, 8C, 45F] 📝
    │   ├── 🏗️ constraints.py [545L, 9C, 39F] 📝
    │   ├── 🏗️ cost_calculator.py [543L, 2C, 18F] 📝
    │   ├── 🏗️ diagnostics.py [352L, 2C, 18F] 📝
    │   ├── 📄 env_config.py [197L, 0C, 13F] 📝
    │   ├── 🏗️ error_handler.py [247L, 6C, 14F] 📝
    │   ├── 🏗️ exceptions.py [177L, 29C, 6F] 📝
    │   ├── 🏗️ file_versioning.py [549L, 2C, 11F] 📝
    │   ├── 🏗️ function_identity.py [194L, 1C, 6F] 📝
    │   ├── 📄 hashing.py [169L, 0C, 6F] 📝
    │   ├── 🏗️ importance.py [513L, 2C, 12F] 📝
    │   ├── 🏗️ incentives.py [404L, 1C, 15F] 📝
    │   ├── 📄 insights.py [412L, 0C, 11F] 📝
    │   ├── 🏗️ langchain_interceptor.py [299L, 1C, 18F] 📝
    │   ├── 🏗️ local_analytics.py [510L, 1C, 12F] 📝
    │   ├── 📄 logging.py [57L, 0C, 2F] 📝
    │   ├── 🏗️ multi_objective.py [504L, 3C, 12F] 📝
    │   ├── 🏗️ numpy_compat.py [93L, 1C, 4F] 📝
    │   ├── 🏗️ optimization_analyzer.py [777L, 1C, 17F] 📝
    │   ├── 🏗️ optimization_logger.py [753L, 1C, 27F] 📝
    │   ├── 🏗️ persistence.py [357L, 2C, 10F] 📝
    │   ├── 🏗️ reproducibility.py [574L, 1C, 21F] 📝
    │   ├── 🏗️ retry.py [609L, 12C, 23F] 📝
    │   └── 🏗️ validation.py [900L, 4C, 31F] 📝
    └── 📦 visualization/ [2f, 634L, 1C, 14F]
        ├── 📋 __init__.py [8L, 0C, 0F] 📝
        └── 🏗️ plots.py [626L, 1C, 14F] 📝
```

### Legend
- `[Nf, NL, NC, NF]` = Files, Lines, Classes, Functions
- 📝 = Has module docstring
- 📦 = Package, 🏗️ = Has classes, 📄 = Module only

## Architectural Analysis

### 🔄 Circular Dependencies

✅ No circular dependencies detected at package level

### 🎯 Hub Modules (High Fan-In)

Modules imported by many others (potential stability concerns):

| Module | Dependents |
|--------|------------|
| `traigent.utils.logging` | 103 |
| `traigent.utils.exceptions` | 52 |
| `traigent.api.types` | 35 |
| `traigent.config.types` | 26 |
| `traigent.evaluators.base` | 25 |
| `traigent.utils.validation` | 20 |
| `traigent.optimizers.base` | 15 |
| `traigent.core.objectives` | 13 |
| `traigent.cloud.models` | 11 |
| `traigent.cloud.backend_client` | 8 |
| `traigent.config.backend_config` | 8 |
| `traigent.invokers.base` | 7 |
| `traigent.utils.retry` | 6 |
| `traigent.core.utils` | 6 |
| `traigent.api.functions` | 5 |

> 💡 **Tip**: Hub modules should be stable and well-tested

### 📊 Layering Analysis

Architectural layers (higher = more abstract):
```
L5: api, cli         (Public Interface)
L4: core             (Core Engine)
L3: cloud, optimizers, evaluators (Services)
L2: integrations, analytics, security, config (Adapters)
L1: utils, storage, telemetry (Foundation)
```

✅ No obvious layering violations detected

## Recommendations

### High Priority

2. **Review hub modules** - 17 modules have high fan-in

### Suggested Next Steps

1. Generate class hierarchy diagrams for core packages
2. Add complexity analysis overlay (cyclomatic complexity)
3. Add test coverage overlay
4. Set up CI to track architecture drift

---

*Generated by Traigent Architecture Analyzer*

## Class Hierarchy Analysis

- **Total Classes**: 624
- **Inheritance Relationships**: 229
- **Abstract Classes**: 18

### Top Classes by Method Count

| Class | Package | Methods | Abstract |
|-------|---------|---------|----------|
| `AuthManager` | traigent.cloud.auth | 65 |  |
| `OptimizationOrchestrator` | traigent.core.orchestrator | 57 |  |
| `BackendIntegratedClient` | traigent.cloud.backend_client | 55 |  |
| `CostOptimizationAI` | traigent.analytics.intelligence | 54 |  |
| `OptimizedFunction` | traigent.core.optimized_function | 50 |  |
| `TraigentCloudClient` | traigent.cloud.client | 46 |  |
| `RemoteServiceRegistry` | traigent.optimizers.service_registry | 40 |  |
| `OptimizationResult` | traigent.api.types | 28 |  |
| `SpecificationGenerator` | traigent.agents.specification_generator | 26 |  |
| `BaseEvaluator` | traigent.evaluators.base | 26 | ✓ |
| `SDKBackendBridge` | traigent.cloud.backend_bridges | 25 |  |
| `DatasetConverter` | traigent.cloud.dataset_converter | 25 |  |
| `ProductionMCPClient` | traigent.cloud.production_mcp_client | 25 |  |
| `ApiOperations` | traigent.cloud.api_operations | 24 |  |
| `LocalEvaluator` | traigent.evaluators.local | 24 |  |

## Complexity Analysis

- **Modules Analyzed**: 223
- **Average Complexity**: 3.85
- **High Complexity Modules**: 20

### High Complexity Modules (Max CC > 15)

| Module | Avg CC | Max CC |
|--------|--------|--------|
| `traigent/evaluators/metrics_tracker.py` | 5.8 | 29 |
| `traigent/integrations/observability/wandb.py` | 4.5 | 29 |
| `traigent/optimizers/optuna_utils.py` | 11.6 | 29 |
| `traigent/cloud/billing.py` | 3.1 | 28 |
| `traigent/core/optimized_function.py` | 5.3 | 28 |
| `traigent/evaluators/local.py` | 7.3 | 28 |
| `traigent/evaluators/metrics.py` | 8.0 | 28 |
| `traigent/optigen_integration.py` | 8.1 | 27 |
| `traigent/core/backend_session_manager.py` | 9.8 | 27 |
| `traigent/utils/optimization_analyzer.py` | 7.6 | 27 |
| `traigent/api/decorators.py` | 5.5 | 26 |
| `traigent/integrations/llms/anthropic_plugin.py` | 5.3 | 26 |
| `traigent/cli/local_commands.py` | 8.8 | 25 |
| `traigent/cloud/backend_synchronizer.py` | 5.2 | 25 |
| `traigent/evaluators/base.py` | 5.8 | 25 |