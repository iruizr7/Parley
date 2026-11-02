-- Keep word/overlap analysis independently of the live transcript schema.
CREATE TABLE meeting_transcription_analysis (
    meeting_id TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
    analysis_json TEXT NOT NULL
);
-- A person may have embeddings from multiple engines. Never compare vectors
-- merely because their dimensions happen to match.
CREATE TABLE voice_profile_embeddings (
    profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    clean_seconds REAL NOT NULL,
    PRIMARY KEY (profile_id, model_id)
);
