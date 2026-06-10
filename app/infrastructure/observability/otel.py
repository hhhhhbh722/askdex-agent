# -*- coding: utf-8 -*-
"""Optional OpenTelemetry integration.

The application can run without OpenTelemetry packages installed. When they are
available, FastAPI requests create standard OTel traces and Agent trace IDs can
reuse the active OTel trace_id.
"""
from __future__ import annotations

from typing import Any

from loguru import logger


def setup_otel(app: Any, service_name: str) -> bool:
    """Instrument FastAPI with OpenTelemetry if the packages are installed."""
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("OpenTelemetry 未启用: {}", exc)
        return False

    try:
        provider = trace.get_tracer_provider()
        if provider.__class__.__name__ == "ProxyTracerProvider":
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        logger.info("✅ OpenTelemetry 已启用 service={}", service_name)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OpenTelemetry 初始化失败: {}", exc)
        return False


def current_otel_trace_id() -> str | None:
    """Return the active OTel trace_id as 32 hex chars, if present."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return None
        return format(ctx.trace_id, "032x")
    except Exception:
        return None


def current_otel_span_id() -> str | None:
    """Return the active OTel span_id as 16 hex chars, if present."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if not ctx or not ctx.is_valid:
            return None
        return format(ctx.span_id, "016x")
    except Exception:
        return None


def set_current_span_attributes(attributes: dict[str, Any]) -> None:
    """Attach attributes to the active OTel span when available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
    except Exception:
        return


def add_current_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Append an event to the active OTel span when available."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        if not span or not span.get_span_context().is_valid:
            return
        span.add_event(name, attributes or {})
    except Exception:
        return
