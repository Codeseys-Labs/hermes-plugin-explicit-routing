"""Explicit per-task provider/model choices for delegate_task, bounded by the owning profile's provider policy.

Runs on stock Hermes (>= 0.23.0) through the public delegation-provider interface: the backend declares one
task parameter, ``route``, and answers ``route_task`` with core route keys that core resolves through the same
credential resolver the ``delegation`` config section uses.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.delegation_provider import DelegationProvider, DelegationRouteError

NAME = 'explicit-routing'
_FIELDS = frozenset({'provider', 'model', 'reasoning_effort'})


def _efforts() -> tuple[str, ...]:
    from hermes_constants import VALID_REASONING_EFFORTS
    return ('none', *VALID_REASONING_EFFORTS)


def _selection(value: Any) -> dict[str, str]:
    """Validate one route object: exact provider and model, optional reasoning_effort, nothing else."""
    if (not isinstance(value, Mapping) or set(value) - _FIELDS
            or not {'provider', 'model'} <= set(value)):
        raise DelegationRouteError('route requires only provider, model and optional reasoning_effort.')
    provider, model = value['provider'], value['model']
    if not isinstance(provider, str) or not provider.strip():
        raise DelegationRouteError('route.provider must be an exact provider ID.')
    if not isinstance(model, str) or not model.strip():
        raise DelegationRouteError('route.model must be a nonempty model ID.')
    result = {'provider': provider, 'model': model}
    if 'reasoning_effort' in value:
        effort = value['reasoning_effort']
        if not isinstance(effort, str) or effort not in _efforts():
            raise DelegationRouteError('route.reasoning_effort is unsupported.')
        result['reasoning_effort'] = effort
    return result


def _allowlist(config: Mapping[str, Any]) -> list[dict[str, str]] | None:
    """``delegation.backends.explicit-routing.allowed``: exact provider/model pairs, or None for no extra limit."""
    backends = config.get('backends', {})
    if not isinstance(backends, Mapping):
        raise DelegationRouteError('delegation.backends must be a mapping.')
    settings = backends.get(NAME, {})
    if not isinstance(settings, Mapping):
        raise DelegationRouteError('explicit-routing settings must be a mapping.')
    if 'allowed' not in settings:
        return None
    raw = settings['allowed']
    if not isinstance(raw, list):
        raise DelegationRouteError('explicit-routing.allowed must be a list of exact provider/model pairs.')
    allowed = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != {'provider', 'model'}:
            raise DelegationRouteError('Each explicit-routing.allowed entry must contain only provider and model.')
        allowed.append(_selection(row))
    return allowed


def _provider_enabled(block: Mapping[str, Any]) -> bool:
    try:
        from hermes_cli.config_providers import is_provider_enabled
    except ImportError:  # older layouts kept it in hermes_cli.config
        from hermes_cli.config import is_provider_enabled
    return bool(is_provider_enabled(dict(block)))


def _check_provider(route: Mapping[str, str], full: Mapping[str, Any]) -> None:
    """Refuse a route to a provider this profile disabled, or an effort the route cannot serve."""
    from hermes_cli.providers import custom_provider_aliases
    from hermes_cli.runtime_provider_custom import named_custom_provider_config_path
    from providers import get_provider_profile

    provider, model = route['provider'].strip().lower(), route['model']
    blocks = full.get('providers')
    entries = list(blocks.items()) if isinstance(blocks, Mapping) else []
    legacy = full.get('custom_providers')
    if isinstance(legacy, list):
        entries.extend((str(block.get('provider_key') or ''), block) for block in legacy if isinstance(block, Mapping))
    for key, block in entries:
        if (isinstance(block, Mapping) and not _provider_enabled(block)
                and provider in custom_provider_aliases(str(block.get('name') or key), str(key))):
            raise DelegationRouteError('The selected provider is disabled in this profile.')
    path = named_custom_provider_config_path(provider, dict(full))
    entry = None
    if path is not None:
        entry = full
        for key in path:
            entry = entry[key]
        if not isinstance(entry, Mapping) or not _provider_enabled(entry):
            raise DelegationRouteError('The selected provider is disabled in this profile.')
    if 'reasoning_effort' in route:
        from agent.reasoning_effort import route_supported_efforts
        profile = get_provider_profile(provider)
        declared = profile.supported_reasoning_efforts(model) if profile is not None else None
        if declared is None and profile is not None and profile.setup_reasoning_efforts is not None:
            declared = profile.setup_reasoning_efforts(model)
        mode = entry.get('api_mode') if entry is not None else getattr(profile, 'api_mode', None)
        supported = declared if declared is not None else route_supported_efforts(provider, model, mode)
        if route['reasoning_effort'] not in supported:
            raise DelegationRouteError('The selected reasoning effort is unsupported on this provider/model route.')


def _fallback_routes(raw: Any) -> list[dict[str, Any]]:
    """Operator-configured fallbacks for explicit routes; an explicit choice never borrows the parent's chain."""
    if raw is None:
        return []
    rows = [raw] if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise DelegationRouteError('Configured fallback_providers must be a route or a list of routes.')
    for row in rows:
        if not isinstance(row, Mapping):
            raise DelegationRouteError('Each configured fallback must be a provider/model route.')
        _selection({key: value for key, value in row.items() if key in _FIELDS})
    from hermes_cli.fallback_config import get_fallback_chain
    return get_fallback_chain({'fallback_providers': [dict(row) for row in rows]}) or []


class ExplicitRouting(DelegationProvider):
    @property
    def name(self) -> str:
        return NAME

    @property
    def display_name(self) -> str:
        return 'Explicit model/provider'

    @property
    def summary(self) -> str:
        return 'Choose an explicit model/provider pair for each task.'

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        # Constant schema: building it never reads credentials or calls providers, so the cached tool
        # prefix of a running conversation does not change.
        return {'route': {'type': 'object', 'additionalProperties': False,
                          'required': ['provider', 'model'], 'properties': {
                              'provider': {'type': 'string'}, 'model': {'type': 'string'},
                              'reasoning_effort': {'type': 'string', 'enum': list(_efforts())},
                          }, 'description': 'Explicit provider/model choice; no keys or endpoints.'}}

    def description(self, config: Mapping[str, Any]) -> str:
        return ('For an explicit choice use tasks[].route with provider, model and optional reasoning_effort. '
                'Existing provider policy and any configured allowlist apply; an omitted or null route uses '
                'ordinary delegation.')

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        if task.get('route') is None:
            # Ordinary delegation: the delegation section itself is the default route.
            return {key: value for key, value in config.items() if key != 'service_tier'}
        selected = _selection(task['route'])
        allowed = _allowlist(config)
        pairs = None if allowed is None else {(row['provider'], row['model']) for row in allowed}
        from hermes_cli.config import load_config_readonly
        full = load_config_readonly()

        def authorize(route: Mapping[str, Any]) -> None:
            if pairs is not None and (route['provider'], route['model']) not in pairs:
                raise DelegationRouteError('This exact provider/model pair is not allowed by explicit-routing policy.')
            _check_provider(route, full)

        authorize(selected)
        fallback = _fallback_routes(config.get('fallback_providers'))
        for value in fallback:
            authorize(value)
        # Only core route keys: core resolves this named provider's credentials and endpoint itself, never the
        # parent's or the delegation section's.
        return {**selected, 'fallback_providers': fallback}


def register(ctx) -> None:
    ctx.register_delegation_provider(ExplicitRouting())
