"""Explicit-routing against stock Hermes core: plugin rules plus core's own route resolution."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def plugin():
    spec = importlib.util.spec_from_file_location('hermes_plugin_explicit_routing', ROOT / '__init__.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def full_config(monkeypatch):
    """The profile's whole config as load_config_readonly() would return it."""
    config = {'providers': {'cpa': {'base_url': 'http://127.0.0.1:9/v1', 'models': {}}}}
    import hermes_cli.config
    monkeypatch.setattr(hermes_cli.config, 'load_config_readonly', lambda: config)
    return config


def test_schema_declares_only_route(plugin):
    params = plugin.ExplicitRouting().task_parameters({})
    assert set(params) == {'route'}
    route = params['route']
    assert route['additionalProperties'] is False
    assert set(route['properties']) == {'provider', 'model', 'reasoning_effort'}
    from agent.delegation_provider import RESERVED_TASK_PARAMETERS
    assert not set(params) & RESERVED_TASK_PARAMETERS


@pytest.mark.parametrize('route', [
    {'provider': 'custom:cpa'},
    {'model': 'gpt-6-astra'},
    {'provider': 'custom:cpa', 'model': 'gpt-6-astra', 'api_key': 'injected'},
    {'provider': 'custom:cpa', 'model': 'gpt-6-astra', 'base_url': 'http://127.0.0.1:9/x'},
    {'provider': ' ', 'model': 'gpt-6-astra'},
    {'provider': 'custom:cpa', 'model': 'gpt-6-astra', 'reasoning_effort': 'super'},
    'custom:cpa/gpt-6-astra',
])
def test_malformed_routes_are_refused(plugin, full_config, route):
    from agent.delegation_provider import DelegationRouteError
    with pytest.raises(DelegationRouteError):
        plugin.ExplicitRouting().route_task({'goal': 'g', 'route': route}, {})


def test_explicit_route_returns_only_core_route_keys(plugin, full_config):
    from agent.delegation_provider import ROUTE_KEYS
    route = plugin.ExplicitRouting().route_task(
        {'goal': 'g', 'route': {'provider': 'custom:cpa', 'model': 'claude-sonnet-5-5'}}, {'model': 'parent-model'})
    assert route == {'provider': 'custom:cpa', 'model': 'claude-sonnet-5-5', 'fallback_providers': []}
    assert set(route) <= ROUTE_KEYS


def test_absent_route_is_ordinary_delegation(plugin, full_config):
    section = {'model': 'm', 'provider': 'p', 'service_tier': 'fast', 'backend': 'explicit-routing'}
    route = plugin.ExplicitRouting().route_task({'goal': 'g'}, section)
    assert route == {'model': 'm', 'provider': 'p', 'backend': 'explicit-routing'}


def test_allowlist_limits_pairs(plugin, full_config):
    from agent.delegation_provider import DelegationRouteError
    section = {'backends': {'explicit-routing': {'allowed': [{'provider': 'custom:cpa', 'model': 'gpt-6-astra'}]}}}
    provider = plugin.ExplicitRouting()
    assert provider.route_task({'route': {'provider': 'custom:cpa', 'model': 'gpt-6-astra'}}, section)['model'] == 'gpt-6-astra'
    with pytest.raises(DelegationRouteError, match='not allowed'):
        provider.route_task({'route': {'provider': 'custom:cpa', 'model': 'claude-sonnet-5-5'}}, section)


def test_disabled_provider_is_refused(plugin, full_config):
    from agent.delegation_provider import DelegationRouteError
    full_config['providers']['cpa']['enabled'] = False
    with pytest.raises(DelegationRouteError, match='disabled'):
        plugin.ExplicitRouting().route_task({'route': {'provider': 'custom:cpa', 'model': 'gpt-6-astra'}}, {})


def test_stock_core_resolves_each_task_on_its_own_route(plugin, full_config, monkeypatch):
    """Stock resolve_task_routes: the backend's schema is offered and each task's route reaches the resolver."""
    from agent.delegation_registry import _registry
    import tools.delegate_tool as delegate_tool
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    _registry.register(plugin.ExplicitRouting())
    seen = []

    def resolve(route, parent):
        seen.append(dict(route))
        return {'provider': route.get('provider'), 'model': route.get('model'), 'base_url': None, 'api_key': None}

    monkeypatch.setattr(delegate_tool, '_resolve_delegation_credentials', resolve)
    section = {'backend': 'explicit-routing'}
    assert 'route' in schema_extension(section)[0]
    routes, error = resolve_task_routes([
        {'goal': 'one', 'route': {'provider': 'custom:cpa', 'model': 'gpt-6-astra'}},
        {'goal': 'two', 'route': {'provider': 'custom:cpa', 'model': 'claude-sonnet-5-5', 'reasoning_effort': 'high'}},
        {'goal': 'bad', 'route': {'provider': 'custom:cpa', 'model': 'x', 'api_key': 'k'}},
    ], section, SimpleNamespace())
    assert error is None
    assert [(r['provider'], r['model']) for r in seen] == [('custom:cpa', 'gpt-6-astra'), ('custom:cpa', 'claude-sonnet-5-5')]
    assert seen[1]['reasoning_effort'] == 'high'
    assert not routes[0].error and not routes[1].error
    assert routes[2].error and 'route requires only' in routes[2].error
