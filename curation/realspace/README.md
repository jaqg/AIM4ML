# curation/realspace — geometry-key identity track (placeholder)

This directory will hold the realspace curation stages: an identity-model
track keyed on molecular *geometry* (structural keys derived from 3D
coordinates) instead of the graph track's canonical-SMILES identity
(CompoundID = MD5 of canonical SMILES, D62).

Motivation: the IQARIS extension targets non-organic species that cannot be
represented as RDKit-sanitized SMILES graphs — diboranes/carboranes (3-center
bonds), noble-gas species, organometallic aggregates (dative bonds, metal
valence), hydrate clusters. The graph track is therefore dispatched *around*
at intake; stages here operate on geometry keys and converge with the graph
track at the shared output stages (extxyz delivery, stats).

Architecture: class-conditional dispatch (D70) — track declared at intake by
the converter/driver; shared machinery lives in top-level `lib/`; genuinely
shared stages in `curation/common/`.

Status: **not yet implemented** — no code. Ingest, geometry-key identity, and
dedup come in a later handoff.
