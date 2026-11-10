"""SSRF-safe fetching of allow-listed web pages for the agent (``web_fetch``).

The URL comes from the planner, which a visitor steers, so every request is
treated as hostile until proven otherwise:

* ``https`` on port 443 only, with no credentials in the URL;
* the host must be on the operator's allow-list (exact names, or
  ``*.example.com`` for its subdomains), and IP-address URLs are refused;
* every address the host resolves to must be public unicast. Loopback, private,
  link-local (cloud metadata at 169.254.169.254), CGNAT, multicast and reserved
  ranges are refused, and so are the IPv6 forms that can embed an IPv4 address
  (IPv4-mapped, 6to4, Teredo, NAT64);
* the connection goes to the address that was checked, with certificate
  verification for the host name. There is no second DNS lookup, so DNS
  rebinding can't swap in an internal address between the check and the
  connection;
* redirects are followed by hand, at most ``max_redirects`` times, and every
  hop is checked like the first URL;
* only uncompressed text responses (HTML, plain text, Markdown, JSON) are read,
  up to ``max_bytes`` and within ``timeout_s`` for the whole fetch.

The tool is disabled unless ``SCRAG_HTTP_TOOL_ALLOWED_HOSTS`` names at least one
host. Fetched text reaches the writer as untrusted evidence, never the planner,
so a page can't make the agent call another tool.
"""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, Field

from ..loaders import html_to_text
from .tools import Tool

USER_AGENT = "scrag-web-fetch/1 (+https://github.com/syed-waleed-ahmed/Self-Correcting-RAG)"
_TEXT_TYPES = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "text",
    "text/markdown": "text",
    "application/json": "text",
}
_REDIRECTS = {301, 302, 303, 307, 308}
_NAT64 = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


class FetchBlocked(Exception):
    """The fetch was refused or failed; the message is safe to show."""


@dataclass(frozen=True)
class RawResponse:
    status: int
    headers: dict[str, str]  # lower-case names
    body: bytes
    truncated: bool = False


# (host, port) -> resolved IP addresses
Resolver = Callable[[str, int], list[str]]
# (address, host, port, target, timeout_s, max_bytes) -> response
Transport = Callable[[str, str, int, str, float, int], RawResponse]


def is_public_address(address: str) -> bool:
    """Whether ``address`` is a public unicast address that's safe to connect to."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])  # drop any IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address):
        embeds_ipv4 = (
            ip.ipv4_mapped is not None or ip.sixtofour is not None or ip.teredo is not None
        )
        if embeds_ipv4 or any(ip in net for net in _NAT64):
            return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


def host_allowed(host: str, allowed: Sequence[str]) -> bool:
    """Exact host names match themselves; ``*.example.com`` matches its subdomains."""
    for pattern in allowed:
        pattern = pattern.strip().rstrip(".").lower()
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


def resolve(host: str, port: int) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS to a fixed, pre-checked address, with SNI and certificate checks for ``host``."""

    def __init__(
        self, host: str, port: int, *, address: str, timeout: float, context: ssl.SSLContext
    ) -> None:
        super().__init__(host, port, timeout=timeout, context=context)
        self._address = address
        self._tls = context

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._tls.wrap_socket(sock, server_hostname=self.host)


def https_transport(
    address: str, host: str, port: int, target: str, timeout: float, max_bytes: int
) -> RawResponse:
    """GET ``target`` from ``host`` over TLS, connecting to ``address`` only."""
    deadline = time.monotonic() + timeout
    conn = _PinnedHTTPSConnection(
        host, port, address=address, timeout=timeout, context=ssl.create_default_context()
    )
    try:
        conn.request(
            "GET",
            target,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html, text/plain;q=0.9, application/json;q=0.8",
                "Accept-Encoding": "identity",
            },
        )
        response = conn.getresponse()
        parts: list[bytes] = []
        size = 0
        while size <= max_bytes:  # a slow server can't stretch the read past the deadline
            if time.monotonic() > deadline:
                raise FetchBlocked("timed out while reading the response")
            part = response.read1(min(65_536, max_bytes + 1 - size))
            if not part:
                break
            parts.append(part)
            size += len(part)
        body = b"".join(parts)
        return RawResponse(
            status=response.status,
            headers={key.lower(): value for key, value in response.getheaders()},
            body=body[:max_bytes],
            truncated=len(body) > max_bytes,
        )
    finally:
        conn.close()


class WebFetcher:
    def __init__(
        self,
        allowed_hosts: Sequence[str],
        *,
        max_bytes: int = 200_000,
        max_chars: int = 6_000,
        timeout_s: float = 8.0,
        max_redirects: int = 3,
        resolver: Resolver = resolve,
        transport: Transport = https_transport,
    ) -> None:
        self.allowed_hosts = [h.strip().rstrip(".").lower() for h in allowed_hosts if h.strip()]
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.timeout_s = timeout_s
        self.max_redirects = max_redirects
        self._resolver = resolver
        self._transport = transport

    def check_url(self, url: str) -> tuple[str, int, str]:
        """``(host, port, request target)`` for an acceptable URL, else FetchBlocked."""
        if len(url) > 2048:
            raise FetchBlocked("the URL is too long")
        if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url.strip()):
            raise FetchBlocked("the URL contains whitespace or control characters")
        parts = urlsplit(url.strip())
        if parts.scheme.lower() != "https":
            raise FetchBlocked("only https:// URLs are allowed")
        if parts.username is not None or parts.password is not None:
            raise FetchBlocked("credentials in URLs are not allowed")
        host = (parts.hostname or "").rstrip(".")
        if not host:
            raise FetchBlocked("the URL has no host")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise FetchBlocked("IP-address URLs are not allowed; use an allow-listed host name")
        try:
            port = parts.port or 443
        except ValueError as exc:
            raise FetchBlocked("the URL has an invalid port") from exc
        if port != 443:
            raise FetchBlocked("only the default https port (443) is allowed")
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise FetchBlocked("the URL has an invalid host name") from exc
        if not host_allowed(host, self.allowed_hosts):
            raise FetchBlocked(f"{host} is not on this server's allow-list")
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return host, port, target

    def fetch(self, url: str) -> dict[str, Any]:
        """Fetch one page as text: ``url``, ``content_type``, ``text``, ``truncated``."""
        deadline = time.monotonic() + self.timeout_s
        current = url.strip()
        for _hop in range(self.max_redirects + 1):
            host, port, target = self.check_url(current)
            try:
                addresses = self._resolver(host, port)
            except OSError as exc:
                raise FetchBlocked(f"could not resolve {host}") from exc
            if not addresses:
                raise FetchBlocked(f"could not resolve {host}")
            # All of them, not just the first: a hostile resolver could mix in one.
            if not all(is_public_address(address) for address in addresses):
                raise FetchBlocked(f"{host} resolves to a non-public address")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FetchBlocked(f"timed out after {self.timeout_s:g}s")
            try:
                response = self._transport(
                    addresses[0], host, port, target, remaining, self.max_bytes
                )
            except (OSError, http.client.HTTPException) as exc:  # includes TLS errors
                raise FetchBlocked(f"the request failed ({type(exc).__name__})") from exc
            if response.status in _REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    raise FetchBlocked("a redirect had no Location header")
                current = urljoin(current, location)
                continue
            return self._read(current, response)
        raise FetchBlocked(f"more than {self.max_redirects} redirects")

    def _read(self, url: str, response: RawResponse) -> dict[str, Any]:
        if response.status != 200:
            raise FetchBlocked(f"the server answered HTTP {response.status}")
        if response.headers.get("content-encoding", "identity").lower() not in ("", "identity"):
            raise FetchBlocked("compressed responses are not accepted")
        content_type = response.headers.get("content-type", "")
        mime = content_type.split(";", 1)[0].strip().lower()
        kind = _TEXT_TYPES.get(mime)
        if kind is None:
            raise FetchBlocked(f"unsupported content type {mime or 'unknown'!r}")
        charset = "utf-8"
        for param in content_type.split(";")[1:]:
            key, _, value = param.partition("=")
            if key.strip().lower() == "charset" and value.strip():
                charset = value.strip().strip('"')
        try:
            text = response.body.decode(charset, errors="replace")
        except LookupError:  # unknown charset name
            text = response.body.decode("utf-8", errors="replace")
        if kind == "html":
            text = html_to_text(text)
        text = text.strip()
        return {
            "url": url,
            "content_type": mime,
            "text": text[: self.max_chars],
            "truncated": response.truncated or len(text) > self.max_chars,
        }


class WebFetchInput(BaseModel):
    url: str = Field(
        min_length=8, max_length=2048, description="An https:// URL on an allow-listed host."
    )


def web_fetch_tool(fetcher: WebFetcher | None) -> Tool:
    """The ``web_fetch`` tool; disabled (but listed) when ``fetcher`` is None."""

    def run(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, WebFetchInput)
        if fetcher is None:
            return {"url": args.url, "error": "web fetching is disabled on this server"}
        try:
            return fetcher.fetch(args.url)
        except FetchBlocked as exc:
            return {"url": args.url, "error": str(exc)}

    hosts = ", ".join(fetcher.allowed_hosts) if fetcher else ""
    return Tool(
        name="web_fetch",
        description=(
            f"Fetch the text of one web page (read-only GET) from an allow-listed host: {hosts}."
            if fetcher
            else "Fetch a web page. Disabled here: no hosts are allow-listed."
        ),
        input_model=WebFetchInput,
        run=run,
        timeout_s=(fetcher.timeout_s + 2.0) if fetcher else 1.0,
        enabled=fetcher is not None and bool(fetcher.allowed_hosts),
    )
