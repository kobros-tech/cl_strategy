"""Compatibility entry point for persistent anonymous routing."""

from __future__ import annotations

from .diagnostics import routing_rank_diagnostics
from .persistent_skill_memory_plugin import (
    PersistentFingerprintSkillMemoryPlugin as _BaseFingerprintPlugin,
)


class PersistentFingerprintSkillMemoryPlugin(_BaseFingerprintPlugin):
    """Persistent router with lightweight defaults and optional diagnostics.

    ``diagnose=False`` keeps route-history retention and diagnostic aggregation
    disabled. Set it to ``True`` when detailed routing records are needed.
    """

    def __init__(
        self,
        *args,
        reverse_epochs: int = 60,
        reverse_batch_size: int = 256,
        diagnose: bool = False,
        **kwargs,
    ) -> None:
        self.diagnose = bool(diagnose)
        self.last_routing_diagnostics: dict = {}
        super().__init__(
            *args,
            reverse_epochs=reverse_epochs,
            reverse_batch_size=reverse_batch_size,
            **kwargs,
        )

    def after_eval_forward(self, strategy, **kwargs) -> None:
        super().after_eval_forward(strategy, **kwargs)
        if self.diagnose:
            self.last_routing_diagnostics = routing_rank_diagnostics(
                self.fingerprint_route_history
            )
        else:
            self.last_routing_diagnostics = {}
            self.fingerprint_route_history.clear()


__all__ = ["PersistentFingerprintSkillMemoryPlugin"]
