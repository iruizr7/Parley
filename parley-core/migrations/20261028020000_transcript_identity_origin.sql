-- Historical labels have unknown provenance. Never infer that they were machine
-- guesses, or silently claim they were explicitly confirmed by a person.
ALTER TABLE transcripts ADD COLUMN identity_origin TEXT NOT NULL DEFAULT 'legacy_unknown'
    CHECK(identity_origin IN ('legacy_unknown','machine','human'));
