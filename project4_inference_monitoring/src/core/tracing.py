import logging
from typing import Optional, Dict, Any
from datetime import datetime
import time

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.warning("LangFuse not available. Using mock tracing.")

logger = logging.getLogger(__name__)


class TracingService:
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None
    ):
        self.enabled = LANGFUSE_AVAILABLE and all([public_key, secret_key])

        if self.enabled:
            self.langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host or "https://cloud.langfuse.com"
            )
            logger.info("LangFuse tracing enabled")
        else:
            self.langfuse = None
            logger.info("Using mock tracing service")

        self.active_traces: Dict[str, Dict[str, Any]] = {}

    def start_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        trace_id = f"trace_{int(time.time() * 1000)}"

        if self.enabled:
            try:
                trace = self.langfuse.trace(
                    name=name,
                    user_id=user_id,
                    metadata=metadata or {}
                )
                self.active_traces[trace_id] = {
                    "trace": trace,
                    "start_time": datetime.now(),
                    "name": name
                }
            except Exception as e:
                logger.error(f"Failed to start LangFuse trace: {e}")
                self.active_traces[trace_id] = {
                    "trace": None,
                    "start_time": datetime.now(),
                    "name": name
                }
        else:
            self.active_traces[trace_id] = {
                "trace": None,
                "start_time": datetime.now(),
                "name": name
            }

        return trace_id

    def end_trace(
        self,
        trace_id: str,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        if trace_id not in self.active_traces:
            logger.warning(f"Trace {trace_id} not found")
            return

        trace_info = self.active_traces[trace_id]
        duration = (datetime.now() - trace_info["start_time"]).total_seconds()

        if self.enabled and trace_info["trace"]:
            try:
                trace_info["trace"].update(
                    success=success,
                    error=error,
                    metadata={"duration_seconds": duration}
                )
            except Exception as e:
                logger.error(f"Failed to update trace: {e}")

        del self.active_traces[trace_id]
        logger.debug(f"Trace {trace_id} ended (duration: {duration:.2f}s)")

    def record_generation(
        self,
        trace_id: str,
        model: str,
        prompt: str,
        completion: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency: float
    ) -> None:
        if trace_id not in self.active_traces:
            return

        if self.enabled and self.active_traces[trace_id].get("trace"):
            try:
                generation = self.active_traces[trace_id]["trace"].generation(
                    name=f"{model}_generation",
                    model=model,
                    input=prompt,
                    output=completion,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    },
                    metadata={
                        "latency_seconds": latency
                    }
                )
                generation.end()
            except Exception as e:
                logger.error(f"Failed to record generation: {e}")

        logger.debug(f"Recorded generation for trace {trace_id}")

    def record_event(
        self,
        trace_id: str,
        event_name: str,
        event_type: str = "custom",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if trace_id not in self.active_traces:
            return

        logger.debug(f"Event '{event_name}' recorded for trace {trace_id}")

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self.active_traces.get(trace_id)

    def flush(self) -> None:
        if self.enabled:
            try:
                self.langfuse.flush()
            except Exception as e:
                logger.error(f"Failed to flush LangFuse: {e}")


_tracing_service: Optional[TracingService] = None


def init_tracing(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None
) -> TracingService:
    global _tracing_service
    _tracing_service = TracingService(public_key, secret_key, host)
    return _tracing_service


def get_tracing_service() -> Optional[TracingService]:
    return _tracing_service
