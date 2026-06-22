"""OpenTelemetry + Langfuse observability for the Signal-MCP stack.

Every MCP server and the agent call `init_tracing()` once at startup.
After that, normal OTel context propagation handles everything:

    - The agent starts a root span per user research session.
    - When the agent calls an MCP tool, the W3C trace context is injected into the MCP `_meta` field (carried through Streamable HTTP).
    - The MCP server reads `_meta` and continues the trace with a child span per tool invocation.
    - Span are exported via OTLP/HTTP to Langfuse's OTel ingestion endpoint.

This produces a single trace tree in Langfuse spanning agent <-> all 4 MCP servers (which are in the servers package), which is the only credible way to debug a multi-server MCP topology.
"""
from __future__ import annotations

import base64, contextlib, os
from typing import Any, Mapping

from opentelemetry import trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.propagate import inject, extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from agent.config import settings


_initialized = False


def init_tracing(component_name: str) -> None:
    """
    Initialize OTel for this process. Idempotent.

    Each MCP server and the agent should call this once at startup.
    `component_name` is appended to the OTel service name so spans are easy to filter in Langfuse (e.g. `signal-mcp.market_data`).
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create({
        "service.name": f"{settings.otel_service_name}.{component_name}",
        "service.version": "0.1.0",
    })

    provider = TracerProvider(resource=resource)

    # Langfuse OTel ingestion uses HTTP Basic auth with public/secret keys
    headers: dict[str, str] = {}
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {token}"

    exporter = OTLPSpanExporter(
        endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces",
        headers=headers,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


# --- MCP _meta <-> OTel context bridge -------------------------------


def inject_trace_into_meta(meta: dict | None = None) -> dict:
    """
    Inject the current OTel context into an MCP `_meta` dict.

    The agent calls this before invoking an MCP tool. The server-side `continue_trace_from_meta` reads it back.
    """
    meta = dict(meta or {})
    carrier: dict[str, str] = {}
    inject(carrier)
    if carrier:
        meta["traceparent"] = carrier.get("traceparent", "")
        if "tracestate" in carrier:
            meta["tracestate"] = carrier["tracestate"]
    return meta


@contextlib.contextmanager
def continue_trace_from_meta(meta: Mapping[str, Any] | None):
    """Server-side context manager: continue a trace from an MCP _meta dict.

    Usage in an MCP server tool:
        @mcp.tool()
        async def my_tool(arg: str, ctx: Context) -> Result:
            with continue_trace_from_meta(ctx.meta):
                with tracer.start_as_current_span("my_tool"):
                    ...
    """
    if not meta:
        yield
        return

    carrier = {}
    if "traceparent" in meta:
        carrier["traceparent"] = meta["traceparent"]
    if "tracestate" in meta:
        carrier["tracestate"] = meta["tracestate"]

    if not carrier:
        yield
        return

    parent_ctx: Context = extract(carrier)
    token = attach(parent_ctx)
    try:
        yield
    finally:
        detach(token)
