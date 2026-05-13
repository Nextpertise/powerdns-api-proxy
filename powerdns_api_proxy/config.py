import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from yaml import safe_load

from powerdns_api_proxy.exceptions import (
    MetricsNotAllowedException,
    NotAuthorizedException,
    ZoneNotAllowedException,
)
from powerdns_api_proxy.logging import logger
from powerdns_api_proxy.models import (
    RRSET,
    ProxyConfig,
    ProxyConfigEnvironment,
    ProxyConfigZone,
    RRSETRequest,
)
from powerdns_api_proxy.pdns import PDNSConnector, handle_pdns_response
from powerdns_api_proxy.utils import check_record_in_regex, check_zones_equal


@lru_cache(maxsize=1)
def load_config(path: Optional[Path] = None) -> ProxyConfig:
    logger.info("Loading config")
    if not path:
        env_path = os.getenv("PROXY_CONFIG_PATH")
        if not env_path:
            raise ValueError("Could not get proxy config path")
        path = Path(env_path)
    with open(path) as f:
        data = safe_load(f)

    config = ProxyConfig(**data)

    return config


def token_defined(config: ProxyConfig, token: str) -> bool:
    sha512 = hashlib.sha512()
    sha512.update(token.encode())
    token_digest = sha512.digest().hex()

    if token_digest in config.token_env_map:
        logger.info(
            f'Authenticated environment "{config.token_env_map[token_digest].name}"'
        )
        return True
    return False


def check_token_defined(config: ProxyConfig, token: str):
    if not token_defined(config, token):
        raise NotAuthorizedException()


def dependency_check_token_defined(
    X_API_Key: str = Header(description="API Key for the proxy."),
):
    check_token_defined(load_config(), X_API_Key)


security = HTTPBasic()


def dependency_metrics_proxy_enabled(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)] = Header(
        description="API Key for the proxy."
    ),
):
    username = credentials.username
    password = credentials.password

    try:
        environment = get_environment_for_token(load_config(), password)
        if not environment.name == username:
            raise NotAuthorizedException()
        if not environment.metrics_proxy:
            raise MetricsNotAllowedException()
    except ValueError:
        raise NotAuthorizedException()


def get_environment_for_token(
    config: ProxyConfig, token: str
) -> ProxyConfigEnvironment:
    """
    Returns:
        ProxyConfigEnvironment: The environment for the given token.
    Raises:
        ValueError: If no environment is found for the given token.
    """
    sha512 = hashlib.sha512()
    sha512.update(token.encode())
    token_digest = sha512.digest().hex()

    if token_digest in config.token_env_map:
        return config.token_env_map[token_digest]

    raise ValueError("Could not find a environment for the given token")


def _zone_account_grants_access(
    environment: ProxyConfigEnvironment, zone_data: dict | None
) -> bool:
    """True if the zone's `account` field matches one configured for the env."""
    if not environment.accounts or not isinstance(zone_data, dict):
        return False
    account = zone_data.get("account") or ""
    return bool(account) and account in environment.accounts


def get_only_pdns_zones_allowed(
    environment: ProxyConfigEnvironment, pdns_zones: list[dict]
) -> list[dict]:
    filtered = []
    if environment.global_read_only:
        return pdns_zones

    for zone in pdns_zones:
        if check_pdns_zone_allowed(environment, zone["name"]):
            filtered.append(zone)
            continue
        if _zone_account_grants_access(environment, zone):
            filtered.append(zone)
    return filtered


def check_pdns_zone_allowed(environment: ProxyConfigEnvironment, zone: str) -> bool:
    """Returns True if zone is allowed in the environment"""
    if environment.global_read_only:
        return True

    try:
        _ = environment.get_zone_if_allowed(zone)
        return True
    except ZoneNotAllowedException:
        return False
    except Exception:
        return False


async def get_zone_account_from_pdns(
    pdns: PDNSConnector, server_id: str, zone_id: str
) -> str | None:
    """
    Fetch a zone's `account` field from PowerDNS.

    The zone id is normalized to canonical form (trailing dot) because
    PowerDNS returns an empty stub for non-canonical names instead of
    looking up the real zone.

    Returns the account string, or None if the zone is missing or the
    upstream response is not parseable as a dict.
    """
    canonical_zone = zone_id if zone_id.endswith(".") else f"{zone_id}."
    resp = await pdns.get(f"/api/v1/servers/{server_id}/zones/{canonical_zone}")
    pdns_response = await handle_pdns_response(resp)
    if not pdns_response.is_success:
        logger.info(
            f"Account lookup for zone '{zone_id}' on server '{server_id}' "
            f"failed: upstream status {pdns_response.status_code}"
        )
        return None
    if not isinstance(pdns_response.data, dict):
        logger.info(
            f"Account lookup for zone '{zone_id}' got non-dict response: "
            f"{type(pdns_response.data).__name__}"
        )
        return None
    account = pdns_response.data.get("account")
    logger.info(f"PowerDNS reports zone '{zone_id}' account = '{account}'")
    return account if account else None


async def resolve_zone_for_environment(
    environment: ProxyConfigEnvironment,
    zone: str,
    pdns: PDNSConnector,
    server_id: str,
    *,
    zone_data: dict | None = None,
) -> ProxyConfigZone:
    """
    Resolve a zone for an environment, allowing two access paths:

    1. Static config: the zone is matched by the environment's `zones` list.
    2. Account-based: the environment declares one or more `accounts`,
       and the zone's `account` field in PowerDNS matches one of them.

    The static path is consulted first. The account path inspects
    `zone_data["account"]` when the caller already has the zone metadata
    in hand (e.g. from proxying `GET /zones/{id}`); otherwise it issues
    an upstream call. Either way access reflects PowerDNS state at
    request time.

    Account-matched zones get RW permissions within the zone (no admin,
    no cryptokeys) via a synthetic ProxyConfigZone.

    Raises ZoneNotAllowedException if neither path grants access.
    """
    try:
        return environment.get_zone_if_allowed(zone)
    except ZoneNotAllowedException:
        pass

    if not environment.accounts:
        raise ZoneNotAllowedException()

    if zone_data is not None:
        if _zone_account_grants_access(environment, zone_data):
            account = zone_data.get("account") if isinstance(zone_data, dict) else None
            logger.info(
                f"Zone '{zone}' granted to environment '{environment.name}' "
                f"via account '{account}' (from proxied response)"
            )
            return ProxyConfigZone(name=zone)
        raise ZoneNotAllowedException()

    logger.info(
        f"Static zones do not allow '{zone}' for environment "
        f"'{environment.name}'; checking accounts {environment.accounts}"
    )
    account = await get_zone_account_from_pdns(pdns, server_id, zone)
    if account and account in environment.accounts:
        logger.info(
            f"Zone '{zone}' granted to environment '{environment.name}' "
            f"via account '{account}'"
        )
        return ProxyConfigZone(name=zone)

    logger.info(
        f"Zone '{zone}' not granted via accounts: PowerDNS account "
        f"'{account}' not in environment.accounts={environment.accounts}"
    )
    raise ZoneNotAllowedException()


def check_pdns_zone_admin(environment: ProxyConfigEnvironment, zone: str) -> bool:
    try:
        env_zone = environment.get_zone_if_allowed(zone)
        return env_zone.admin
    except ZoneNotAllowedException:
        pass
    except Exception:
        pass
    return False


def check_pdns_search_allowed(
    environment: ProxyConfigEnvironment, query: str, object_type: str
) -> bool:
    if environment.global_search:
        return True
    return False


def check_rrset_allowed(zone: ProxyConfigZone, rrset: RRSET) -> bool:
    if zone.read_only:
        return False

    if zone.all_records:
        return True

    if not zone.regex and not rrset["name"].rstrip(".").endswith(zone.name.rstrip(".")):
        logger.debug("RRSET not allowed, because zone does not match")
        return False

    for record in zone.records:
        if check_zones_equal(rrset["name"], record):
            return True

    for regex in zone.regex_records:
        if check_record_in_regex(rrset["name"], regex):
            return True

    if check_acme_record_allowed(zone, rrset):
        return True

    return False


def check_acme_record_allowed(zone: ProxyConfigZone, rrset: RRSET) -> bool:
    if zone.all_records:
        logger.debug("ACME challenge allowed, because all records are allowed")
        return True

    if not zone.services.acme:
        logger.info("Service ACME is not activated")
        return False

    for record in zone.records:
        if check_zones_equal(f"_acme-challenge.{record}", rrset["name"]):
            logger.info(f"ACME challenge for record {record} is allowed")
            return True

    return False


def check_pdns_cryptokeys_allowed(
    environment: ProxyConfigEnvironment, zone: str
) -> bool:
    if environment.global_cryptokeys:
        return True

    try:
        return environment.get_zone_if_allowed(zone).cryptokeys
    except ZoneNotAllowedException:
        pass

    return False


def check_pdns_tsigkeys_allowed(environment: ProxyConfigEnvironment) -> bool:
    if environment.global_tsigkeys:
        return True
    return False


def check_pdns_config_allowed(environment: ProxyConfigEnvironment) -> bool:
    return environment.global_config


def check_pdns_statistics_allowed(environment: ProxyConfigEnvironment) -> bool:
    return environment.global_statistics


def ensure_rrsets_request_allowed(zone: ProxyConfigZone, request: RRSETRequest) -> bool:
    """Raises HTTPException if RRSET is not allowed"""
    if zone.read_only:
        logger.info("RRSET update not allowed with read only token")
        raise HTTPException(403, "RRSET update not allowed with read only token")
    for rrset in request["rrsets"]:
        if not check_rrset_allowed(zone, rrset):
            logger.info(f"RRSET {rrset['name']} not allowed in zone {zone.name}")
            raise HTTPException(403, f"RRSET {rrset['name']} not allowed")
        logger.info(f"RRSET {rrset['name']} allowed")
    return True
