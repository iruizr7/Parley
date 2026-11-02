-- Additive pipeline history and evidence. Historical artifacts remain unknown.
CREATE TABLE processing_runs (
 id TEXT PRIMARY KEY, meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
 operation TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
 outcome TEXT NOT NULL, requested_json TEXT NOT NULL, resolved_json TEXT NOT NULL,
 input_json TEXT NOT NULL, result_json TEXT, error TEXT
);
CREATE TABLE meeting_processing_state (
 meeting_id TEXT PRIMARY KEY REFERENCES meetings(id) ON DELETE CASCADE,
 current_run_id TEXT REFERENCES processing_runs(id), group_version TEXT
);
CREATE TABLE identification_runs (
 id TEXT PRIMARY KEY, meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
 processing_run_id TEXT, group_version TEXT, created_at TEXT NOT NULL,
 policy_version TEXT NOT NULL, request_json TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE TABLE speaker_identity_decisions (
 meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
 group_version TEXT NOT NULL, speaker TEXT NOT NULL, profile_id TEXT,
 confirmed INTEGER NOT NULL DEFAULT 0, identification_run_id TEXT, updated_at TEXT NOT NULL,
 PRIMARY KEY(meeting_id,group_version,speaker)
);
CREATE TABLE profile_enrollments (
 profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
 space_id TEXT NOT NULL, revision TEXT NOT NULL, status TEXT NOT NULL,
 embedding_json TEXT, clean_seconds REAL NOT NULL DEFAULT 0,
 updated_at TEXT NOT NULL, extraction_version TEXT NOT NULL,
 space_json TEXT NOT NULL DEFAULT '{}',
 PRIMARY KEY(profile_id,space_id)
);
CREATE TABLE enrollment_samples (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES voice_profiles(id) ON DELETE CASCADE,
 space_id TEXT NOT NULL, source_meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
 audio_sha256 TEXT, channel INTEGER NOT NULL DEFAULT 0, intervals_json TEXT NOT NULL,
 source_run_id TEXT, confirmation_json TEXT NOT NULL, embedding_json TEXT,
 clean_seconds REAL NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX enrollment_samples_profile_space ON enrollment_samples(profile_id,space_id);
-- Legacy vectors have no invented audio source or reproducibility claim.
INSERT INTO profile_enrollments(profile_id,space_id,revision,status,embedding_json,clean_seconds,updated_at,extraction_version)
 SELECT profile_id,'legacy:'||model_id,lower(hex(randomblob(16))),'legacy_unknown',embedding_json,clean_seconds,datetime('now'),'unknown'
 FROM voice_profile_embeddings;
-- Only current groups' extraction evidence is cached. Historical run manifests
-- retain its provenance, never biometric vectors solely for history.
CREATE TABLE speaker_group_embeddings (
 meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
 group_version TEXT NOT NULL,speaker TEXT NOT NULL,space_id TEXT NOT NULL,
 embedding_json TEXT NOT NULL,provenance_json TEXT NOT NULL,
 PRIMARY KEY(meeting_id,group_version,speaker,space_id)
);
CREATE TABLE identity_decision_events (
 id TEXT PRIMARY KEY,meeting_id TEXT REFERENCES meetings(id) ON DELETE SET NULL,
 group_version TEXT,speaker TEXT,profile_id TEXT,decision TEXT NOT NULL,
 created_at TEXT NOT NULL,provenance_json TEXT NOT NULL
);
-- Existing native blobs have no recoverable checkpoint/source metadata.
INSERT INTO profile_enrollments(profile_id,space_id,revision,status,embedding_json,clean_seconds,updated_at,extraction_version)
 SELECT id,'legacy:campplus:unknown',lower(hex(randomblob(16))),'legacy_unknown',NULL,0,datetime('now'),'unknown'
 FROM voice_profiles v WHERE embedding_dim>0 AND NOT EXISTS(SELECT 1 FROM profile_enrollments e WHERE e.profile_id=v.id AND e.space_id LIKE '%campplus%');

CREATE TABLE enrollment_versions (
 profile_id TEXT NOT NULL,space_id TEXT NOT NULL,revision TEXT NOT NULL,
 created_at TEXT NOT NULL,summary_json TEXT NOT NULL,
 PRIMARY KEY(profile_id,space_id,revision)
);
