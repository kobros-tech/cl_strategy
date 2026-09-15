# Persistent class behavioral fingerprints

This branch starts the implementation on top of `v0.1.5-best-skill-routing`.

The implementation stores reference behavior per canonical class, versions it
by mutable skill slot, invalidates all classes of a changed skill after REUSE,
and uses class-aligned cosine matching for anonymous evaluation routing.

Reference output vectors are keyed by global class IDs rather than fixed
classifier positions, so classifier-head growth does not invalidate the
representation merely by adding new output columns.

`PersistentFingerprintSkillMemoryPlugin` is an opt-in extension while the
existing `SkillMemoryPlugin` remains unchanged. This keeps the current routing
implementation available while the behavioral fingerprint path is validated.
