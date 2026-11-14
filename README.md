# explicit-routing — Hermes delegation backend

Lets each `delegate_task` task name the exact provider and model it runs on:

```python
delegate_task(tasks=[
    {"goal": "Summarize the diff", "route": {"provider": "custom:cpa", "model": "gpt-6-astra"}},
    {"goal": "Review the summary", "route": {"provider": "custom:cpa", "model": "claude-sonnet-5-5",
                                              "reasoning_effort": "high"}},
])
```

A task without `route` (or with `route: null`) uses ordinary delegation (the `delegation` config section).
Routes carry no keys or endpoints: Hermes core resolves the named provider's credentials the same way it does
for `delegation.provider`.

Works on stock Hermes Agent 0.23.x through the public delegation-provider interface. No core patches.

## Install

```sh
hermes plugins install <git-url-of-this-repo> --ref <40-character-commit-sha> --enable
```

Then select the backend in each profile that should offer it (`config.yaml`):

```yaml
delegation:
  backend: explicit-routing
  # optional: limit routes to exact pairs
  backends:
    explicit-routing:
      allowed:
        - {provider: custom:cpa, model: gpt-6-astra}
        - {provider: custom:cpa, model: claude-sonnet-5-5}
```

Running sessions keep the tool schema they started with: use `/new` (or `/delegation explicit-routing --now`)
after enabling.

## Rules

- `route` must contain `provider` and `model`, optionally `reasoning_effort`; anything else is refused.
- A provider disabled in the profile (`enabled: false`) is refused, as is an effort the route cannot serve.
- With `allowed` set, only those exact pairs are accepted.
- An explicit route never inherits the parent's fallback chain; `delegation.fallback_providers` (if set) is
  checked against the same rules and used instead.
- Model aliases are not expanded: name the exact model ID the provider serves.

## Tests

Run against a Hermes source tree on `PYTHONPATH` with its dependencies installed:

```sh
PYTHONPATH=/path/to/hermes-agent python -m pytest tests
```
