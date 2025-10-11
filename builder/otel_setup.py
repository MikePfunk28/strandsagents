# builder/otel_setup.py
import logging
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor


def setup_otel(service_name: str = "strands-agent-factory"):
    # Traces (console by default; swap with OTLP/Jaeger as needed)
    resource = Resource.create({"service.name": service_name})
    tp = TracerProvider(resource=resource)
    tp.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(tp)

    # Logs → OTLP if OTLP_ENDPOINT is set, else standard logging
    logger_provider = LoggerProvider(resource=resource)
    if os.getenv("OTLP_ENDPOINT"):
        logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(
                endpoint=os.environ["OTLP_ENDPOINT"]))
        )
    handler = LoggingHandler(
        level=logging.INFO, logger_provider=logger_provider)
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    return trace.get_tracer(service_name)
