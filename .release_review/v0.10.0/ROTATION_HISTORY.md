/home/nimrodbu/Traigent_enterprise/Traigent/.venv/lib/python3.12/site-packages/instructor/providers/gemini/client.py:6: FutureWarning:

All support for the `google.generativeai` package has ended. It will no longer be receiving
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
# Rotation Schedule: v0.10.0 (Round 2)

Generated: 2026-02-14T01:03:44.612481Z

| Category | Primary | Secondary | Spot-Check |
|----------|---------|-----------|------------|
| Security/Core | GPT-5.3 | Gemini 3 Pro | Claude Opus 4.6 |
| Integrations | Gemini 3 Pro | Claude Opus 4.6 | GPT-5.3 |
| Packaging/CI | Claude Opus 4.6 | GPT-5.3 | Gemini 3 Pro |
| Docs/Examples | GPT-5.3 | Gemini 3 Pro | Claude Opus 4.6 |

Save to history? [y/N]: Traceback (most recent call last):
  File "/home/nimrodbu/Traigent_enterprise/Traigent/.release_review/automation/rotation_scheduler.py", line 524, in <module>
    main()
  File "/home/nimrodbu/Traigent_enterprise/Traigent/.release_review/automation/rotation_scheduler.py", line 460, in main
    save = input("Save to history? [y/N]: ").strip().lower()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
EOFError: EOF when reading a line
