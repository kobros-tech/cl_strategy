"""Class-level probe-based Skill Memory for Avalanche."""

from .cl.decision import find_best_skill
from .cl.skill_memory_plugin import SkillMemoryPlugin
from .cl.skill_registry import ClassRecord, ExperienceClassMap, SkillMemory
from .evaluation.behavior import (
    BehaviorFingerprintCache,
    ClassBehaviorRecord,
    compare_binary_behavior,
    identify_binary_behavior,
    reverse_engineer_scores_from_weights,
    reverse_engineer_y,
    reverse_engineer_y_from_weights,
)
from .evaluation.diagnostics import class_index_alignment_report
from .evaluation.fingerprint_routing import PersistentFingerprintSkillMemoryPlugin
from .evaluation.reverse_engineering import CandidateParameters, NormalMLReverseEngineer
from .evaluation.routing import RoutingResult, find_best_routing_skill

__all__ = [
    "BehaviorFingerprintCache",
    "CandidateParameters",
    "ClassRecord",
    "ClassBehaviorRecord",
    "ExperienceClassMap",
    "RoutingResult",
    "SkillMemory",
    "SkillMemoryPlugin",
    "PersistentFingerprintSkillMemoryPlugin",
    "NormalMLReverseEngineer",
    "class_index_alignment_report",
    "compare_binary_behavior",
    "identify_binary_behavior",
    "reverse_engineer_scores_from_weights",
    "reverse_engineer_y",
    "reverse_engineer_y_from_weights",
    "find_best_routing_skill",
    "find_best_skill",
]
