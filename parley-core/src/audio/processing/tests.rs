use super::*;
use sqlx::SqlitePool;
async fn pool() -> SqlitePool {
    let p = sqlx::sqlite::SqlitePoolOptions::new()
        .max_connections(1)
        .connect("sqlite::memory:")
        .await
        .unwrap();
    sqlx::migrate!("./migrations").run(&p).await.unwrap();
    p
}
fn analysis() -> types::Analysis {
    serde_json::from_value(serde_json::json!({"version":1,"duration":12.,"language":"en","segments":[{"start":1.,"end":5.,"text":"Fictional example.","speaker":"Speaker 1"}],"words":[],"turns":[{"start":0.,"end":10.,"speaker":"Speaker 1"}],"overlaps":[],"embeddings":[],"settings":{"request_id":"example-job","config":{"speech_engine":"whisper_cpp"}},"provenance":{"devices":{"speech":"cpu"}}})).unwrap()
}
async fn meeting(p: &SqlitePool, a: &types::Analysis) -> String {
    crate::database::repositories::transcript::TranscriptsRepository::save_transcript_with_analysis(
        p,
        "Fictional meeting",
        &a.transcript_segments(None).await.unwrap(),
        None,
        Some(a),
    )
    .await
    .unwrap()
}
#[tokio::test]
async fn successful_and_failed_runs_are_immutable_and_keep_current_pointer() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    assert_eq!(
        runs::current_run(&p, &m).await.unwrap().as_deref(),
        Some("example-job")
    );
    runs::record_failure(
        &p,
        &m,
        "process",
        &serde_json::json!({"speech_engine":"faster_whisper"}),
        "Fictional failure",
        false,
    )
    .await
    .unwrap();
    let history = runs::list_runs(&p, &m).await.unwrap();
    assert_eq!(history.len(), 2);
    assert!(history.iter().any(|r| r.outcome == "failed"));
    assert_eq!(
        runs::current_run(&p, &m).await.unwrap().as_deref(),
        Some("example-job")
    );
    assert_eq!(
        runs::load(&p, &m).await.unwrap().unwrap().segments[0].text,
        "Fictional example."
    );
}
#[tokio::test]
async fn cancelled_publish_keeps_previous_transcript_and_history() {
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    a.settings["request_id"] = "cancelled-job".into();
    a.segments[0].text = "Replacement".into();
    let c = std::sync::atomic::AtomicBool::new(true);
    let segments = a.transcript_segments(None).await.unwrap();
    assert!(crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis_cancel(&p,&m,&segments,Some(&a),Some(&c)).await.is_err());
    assert_eq!(
        runs::load(&p, &m).await.unwrap().unwrap().segments[0].text,
        "Fictional example."
    );
    assert_eq!(runs::list_runs(&p, &m).await.unwrap().len(), 1);
}
#[tokio::test]
async fn migration_preserves_legacy_defaults_and_target_setup_without_io() {
    let p = pool().await;
    crate::database::repositories::setting::SettingsRepository::save_transcript_config(
        &p,
        "localWhisper",
        "small",
    )
    .await
    .unwrap();
    sqlx::query("INSERT INTO app_settings VALUES('ui_config',?,datetime('now'))").bind(serde_json::json!({"offlinePython":"custom-python","offlineAssets":"model-cache","offlineDevice":"cuda"}).to_string()).execute(&p).await.unwrap();
    let c = config::load_config(&p).await.unwrap();
    assert_eq!(c.speech_model, "small");
    assert!(
        config::validate_config(&c).is_err(),
        "unsupported legacy choice needs explicit selection, never substitution"
    );
    let target = config::get_target(&p, "local").await.unwrap();
    assert_eq!(target.python.as_deref(), Some("custom-python"));
    assert_eq!(
        target.device, "auto",
        "legacy Python device must never force the native pipeline off its GPU"
    );
    assert_eq!(target.legacy_python_device.as_deref(), Some("cuda"));
    let mut python = c;
    python.speech_engine = config::SpeechEngine::FasterWhisper;
    python.speech_model = "large-v3-turbo-fp16".into();
    assert_eq!(
        config::get_target_for_config(&p, &python)
            .await
            .unwrap()
            .device,
        "cuda"
    );
}
#[test]
fn clean_evidence_excludes_overlap_and_keeps_actual_intervals() {
    let mut a = analysis();
    a.overlaps.push(types::Overlap {
        start: 4.,
        end: 6.,
        speakers: vec!["Speaker 1".into(), "Speaker 2".into()],
    });
    let spans = service::clean_intervals(&a);
    assert_eq!(spans.len(), 2);
    assert_eq!((spans[0].start, spans[0].end), (0., 4.));
    assert_eq!((spans[1].start, spans[1].end), (6., 10.));
}
#[tokio::test]
async fn evidence_removal_rebuilds_without_removed_vector() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional person",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let space = types::EmbeddingSpace {
        id: "example-checkpoint:sha256:fixture:preprocessing-v1".into(),
        model: "example".into(),
        revision: "fixture".into(),
        preprocessing: "mono16k".into(),
        normalization: "l2".into(),
        aggregation: "mean".into(),
        extraction_version: "test-v1".into(),
        dimension: 2,
    };
    let mut tx = p.begin().await.unwrap();
    for (index, vector) in [vec![1., 0.], vec![0., 1.]].into_iter().enumerate() {
        let e = types::SpeakerEmbedding {
            speaker: "Speaker 1".into(),
            model_id: "example".into(),
            vector,
            clean_seconds: 4.,
            sample_count: 1,
            embedding_space: Some(space.clone()),
            source_intervals: vec![types::SourceInterval {
                speaker: "Speaker 1".into(),
                start: 1. + 5. * index as f64,
                end: 5. + 5. * index as f64,
                channel: 0,
            }],
        };
        enrollment::store_evidence_tx(&mut tx, &profile, &m, Some("example-job"), None, &e)
            .await
            .unwrap();
    }
    tx.commit().await.unwrap();
    let samples = enrollment::list_samples(&p, &profile, &space.id)
        .await
        .unwrap();
    assert_eq!(samples.len(), 2);
    enrollment::remove_sample(&p, &samples[0].id).await.unwrap();
    let centroid: String =
        sqlx::query_scalar("SELECT embedding_json FROM profile_enrollments WHERE profile_id=?")
            .bind(&profile)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<f32>>(&centroid).unwrap(),
        vec![0., 1.]
    );
    enrollment::remove_sample(&p, &samples[1].id).await.unwrap();
    assert_eq!(
        enrollment::list_enrollments(&p, &profile).await.unwrap()[0].status,
        "insufficient_clean_evidence"
    );
    let vector: Option<String> =
        sqlx::query_scalar("SELECT embedding_json FROM profile_enrollments WHERE profile_id=?")
            .bind(&profile)
            .fetch_one(&p)
            .await
            .unwrap();
    assert!(vector.is_none());
}
#[test]
fn exports_redact_connections_and_biometrics_recursively() {
    let value = runs::redact_metadata(
        &serde_json::json!({"target":{"name":"Render target","ssh_host":"private-alias","worker_path":"private-dir"},"embeddings":[{"vector":[0.,1.],"embedding_space":{"id":"example-space"}}],"resolved_policy":{"expected_profile_ids":["fictional-private-attendee"],"candidate_contexts":[{"profile_id":"fictional-private-profile"}]}}),
    );
    assert!(value["target"].get("ssh_host").is_none());
    assert!(value["embeddings"][0].get("vector").is_none());
    assert_eq!(value["target"]["name"], "Render target");
    assert!(value.get("resolved_policy").is_none());
}

#[tokio::test]
async fn identification_off_preserves_legacy_human_names_without_asr_or_target() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    sqlx::query("UPDATE transcripts SET speaker='Fictional confirmed name',identity_origin='legacy_unknown' WHERE meeting_id=?")
        .bind(&m)
        .execute(&p)
        .await
        .unwrap();
    let c = std::sync::atomic::AtomicBool::new(false);
    let r = service::identify_speakers(
        &p,
        &m,
        service::IdentificationOptions {
            identification: config::Identification::Off,
            target_id: "missing-target".into(),
            replace_confirmed: false,
        },
        &c,
        |_, _, _| panic!("identification off cannot start inference"),
    )
    .await
    .unwrap();
    assert!(r.suggestions[0].confirmed);
    let rows: Vec<(String, f64, f64)> = sqlx::query_as(
        "SELECT transcript,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_all(&p)
    .await
    .unwrap();
    assert_eq!(rows, vec![("Fictional example.".into(), 1., 5.)]);
    let name: String = sqlx::query_scalar("SELECT speaker FROM transcripts WHERE meeting_id=?")
        .bind(&m)
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(name, "Fictional confirmed name");
}
#[tokio::test]
async fn exact_embedding_spaces_isolate_identical_dimensions() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional person",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let mut e:types::SpeakerEmbedding=serde_json::from_value(serde_json::json!({"speaker":"Speaker 1","model_id":"example","vector":[1.,0.],"clean_seconds":4.,"embedding_space":{"id":"checkpoint-a:preprocess-v1","model":"example","revision":"a","preprocessing":"mono16k","normalization":"l2","aggregation":"mean","extraction_version":"v1","dimension":2},"source_intervals":[{"speaker":"Speaker 1","start":1.,"end":5.,"channel":0}]})).unwrap();
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, None, &e)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    assert_eq!(
        identification::match_profiles_for_meeting(&p, &m, &[e.clone()])
            .await
            .unwrap()["Speaker 1"]
            .0,
        profile
    );
    e.embedding_space.as_mut().unwrap().id = "checkpoint-b:preprocess-v1".into();
    assert!(identification::match_profiles_for_meeting(&p, &m, &[e])
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn explicit_acceptance_protects_current_group_and_never_enrolls() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional confirmed person",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    assert!(
        service::confirm_identity_suggestion(&p, &m, "stale-group", "Speaker 1", &profile)
            .await
            .is_err()
    );
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &profile)
        .await
        .unwrap();
    let suggestions = service::current_identity_suggestions(&p, &m).await.unwrap();
    assert!(suggestions[0].confirmed);
    assert_eq!(suggestions[0].profile_id.as_deref(), Some(profile.as_str()));
    assert!(enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .is_empty());
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM identity_decision_events")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(count, 1);
}

#[tokio::test]
async fn source_playback_uses_actual_intervals_and_unavailable_audio_retains_vector() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let folder = tempfile::tempdir().unwrap();
    let audio = folder.path().join("audio.wav");
    std::fs::write(&audio, b"fictional audio identity fixture").unwrap();
    sqlx::query("UPDATE meetings SET folder_path=? WHERE id=?")
        .bind(folder.path().to_string_lossy().as_ref())
        .bind(&m)
        .execute(&p)
        .await
        .unwrap();
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional source",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let e:types::SpeakerEmbedding=serde_json::from_value(serde_json::json!({"speaker":"Speaker 1","model_id":"fixture","vector":[1.,0.],"clean_seconds":4.,"embedding_space":{"id":"fixture:space-v1","model":"fixture","revision":"fixture","preprocessing":"mono16k","normalization":"l2","aggregation":"mean","extraction_version":"v1","dimension":2},"source_intervals":[{"speaker":"Speaker 1","start":1.,"end":5.,"channel":0}]})).unwrap();
    let hash = runs::audio_hash(audio.clone()).await.unwrap();
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, Some("example-job"), Some(&hash), &e)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let sample = enrollment::list_samples(&p, &profile, "fixture:space-v1")
        .await
        .unwrap()
        .remove(0);
    let playback = enrollment::source_playback(&p, &sample.id).await.unwrap();
    assert_eq!((playback[0].start, playback[0].end), (1., 5.));
    assert!(playback[0].downmix);
    std::fs::write(&audio, b"changed fictional audio").unwrap();
    assert!(enrollment::source_playback(&p, &sample.id).await.is_err());
    std::fs::remove_file(audio).unwrap();
    assert!(
        !enrollment::list_samples(&p, &profile, "fixture:space-v1")
            .await
            .unwrap()[0]
            .available
    );
    assert_eq!(
        enrollment::list_enrollments(&p, &profile).await.unwrap()[0].status,
        "ready"
    );
}

#[tokio::test]
async fn concurrent_target_edits_preserve_both_targets() {
    let p = pool().await;
    let a = config::ProcessingTarget {
        id: "fixture-a".into(),
        name: "Fictional A".into(),
        ..Default::default()
    };
    let b = config::ProcessingTarget {
        id: "fixture-b".into(),
        name: "Fictional B".into(),
        ..Default::default()
    };
    let (ar, br) = tokio::join!(config::save_target(&p, &a), config::save_target(&p, &b));
    ar.unwrap();
    br.unwrap();
    let targets = config::list_targets(&p).await.unwrap();
    assert!(targets.iter().any(|t| t.id == a.id));
    assert!(targets.iter().any(|t| t.id == b.id));
}

#[test]
fn partial_embedding_cache_still_extracts_missing_clean_groups() {
    let mut a = analysis();
    a.turns.push(types::TimedSpeaker {
        start: 10.,
        end: 14.,
        speaker: "Speaker 2".into(),
    });
    a.duration = 14.;
    a.embeddings.push(types::SpeakerEmbedding {
        speaker: "Speaker 1".into(),
        model_id: config::NATIVE_EMBEDDING.into(),
        vector: vec![1., 0.],
        clean_seconds: 4.,
        sample_count: 1,
        embedding_space: Some(types::EmbeddingSpace {
            id: speaker_core::campplus_space_id("fictional-checkpoint-hash"),
            model: "campplus".into(),
            revision: "fictional-checkpoint-hash".into(),
            preprocessing: "pcm16k-mono-ffmpeg-v1".into(),
            normalization: "l2".into(),
            aggregation: "longest-three-trimmed-10s-v1".into(),
            extraction_version: "parley-native-v2".into(),
            dimension: 2,
        }),
        source_intervals: vec![types::SourceInterval {
            speaker: "Speaker 1".into(),
            start: 1.,
            end: 5.,
            channel: 0,
        }],
    });
    let needed = service::missing_embedding_intervals(&a, "campplus");
    assert_eq!(needed.len(), 1);
    assert_eq!(needed[0].speaker, "Speaker 2");
}

#[tokio::test]
async fn offline_campplus_enrollment_never_relabels_legacy_live_embedding_space() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional live voice",
        None,
        &[0., 1., 0.],
        3,
    )
    .await
    .unwrap();
    let e:types::SpeakerEmbedding=serde_json::from_value(serde_json::json!({"speaker":"Speaker 1","model_id":"campplus","vector":[1.,0.],"clean_seconds":4.,"embedding_space":{"id":"campplus:sha256:fictional:offline-v2","model":"campplus","revision":"fictional","preprocessing":"pcm16k-mono-ffmpeg-v1","normalization":"l2","aggregation":"longest-three-trimmed-10s-v1","extraction_version":"parley-native-v2","dimension":2},"source_intervals":[{"speaker":"Speaker 1","start":1.,"end":5.,"channel":0}]})).unwrap();
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, None, &e)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let saved = crate::database::repositories::voice_profile::VoiceProfilesRepository::get_by_id(
        &p, &profile,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(
        crate::database::repositories::voice_profile::bytes_to_floats(&saved.embedding).unwrap(),
        vec![0., 1., 0.]
    );
    enrollment::delete_enrollment(&p, &profile, &e.embedding_space.unwrap().id)
        .await
        .unwrap();
    let saved = crate::database::repositories::voice_profile::VoiceProfilesRepository::get_by_id(
        &p, &profile,
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(saved.embedding_dim, 3);
}

#[tokio::test]
async fn automatic_suggestion_cannot_enroll_until_explicit_source_confirmation() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional suggested person",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    sqlx::query("UPDATE transcripts SET speaker='Fictional suggested person',voice_profile_id=? WHERE meeting_id=?").bind(&profile).bind(&m).execute(&p).await.unwrap();
    sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,'example-job','Speaker 1',?,0,NULL,datetime('now'))").bind(&m).bind(&profile).execute(&p).await.unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let result = enrollment::enroll_from_meeting(
        &p,
        &profile,
        &m,
        "Fictional suggested person",
        config::Identification::Campplus,
        "missing-target",
        &cancel,
        |_, _, _| panic!("unconfirmed suggestions must not start inference"),
    )
    .await;
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Confirm this speaker"));
    assert!(enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .is_empty());
    assert!(enrollment::confirm_enrollment_source(
        &p,
        &profile,
        &m,
        "Fictional suggested person",
        Some("stale-group")
    )
    .await
    .is_err());
    enrollment::confirm_enrollment_source(
        &p,
        &profile,
        &m,
        "Fictional suggested person",
        Some("example-job"),
    )
    .await
    .unwrap();
    let confirmed: bool =
        sqlx::query_scalar("SELECT confirmed FROM speaker_identity_decisions WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert!(confirmed);
    assert!(
        enrollment::list_enrollments(&p, &profile)
            .await
            .unwrap()
            .is_empty(),
        "confirmation does not itself learn voice evidence"
    );
}

#[test]
fn enrollment_unions_contiguous_confirmed_rows_without_bridging_gaps_or_other_voices() {
    let mut a = analysis();
    a.turns = vec![types::TimedSpeaker {
        speaker: "Speaker 1".into(),
        start: 0.,
        end: 6.,
    }];
    a.duration = 6.;
    let contiguous = [(0., 2.), (2., 4.), (4., 6.)];
    let clean = enrollment::confirmed_clean_intervals(
        &a,
        &contiguous,
        &[],
        &std::collections::BTreeSet::from(["Speaker 1".into()]),
    );
    assert_eq!(clean.len(), 1);
    assert_eq!((clean[0].start, clean[0].end), (0., 6.));
    assert_eq!(clean[0].speaker, "Speaker 1");
    assert!(enrollment::confirmed_clean_intervals(
        &a,
        &[(0., 2.), (3., 5.)],
        &[],
        &std::collections::BTreeSet::from(["Speaker 1".into()])
    )
    .is_empty());
    assert!(enrollment::confirmed_clean_intervals(
        &a,
        &contiguous,
        &[(2., 4.)],
        &std::collections::BTreeSet::from(["Speaker 1".into()])
    )
    .is_empty());
    assert_eq!(
        enrollment::merge_confirmed_intervals(&[(0., 2.), (2., 4.), (5., 6.)]),
        vec![(0., 4.), (5., 6.)]
    );
}

#[test]
fn long_primary_asr_segment_never_authorizes_secondary_voice_turns() {
    let mut a = analysis();
    a.duration = 12.;
    a.segments = vec![types::Segment {
        start: 0.,
        end: 12.,
        text: "Fictional primary row.".into(),
        speaker: Some("Speaker 1".into()),
        overlapping_speakers: vec![],
    }];
    a.turns = vec![
        types::TimedSpeaker {
            speaker: "Speaker 1".into(),
            start: 0.,
            end: 4.,
        },
        types::TimedSpeaker {
            speaker: "Speaker 4".into(),
            start: 4.,
            end: 8.,
        },
        types::TimedSpeaker {
            speaker: "Speaker 1".into(),
            start: 8.,
            end: 12.,
        },
    ];
    let spans = [(0., 12.)];
    let groups = enrollment::source_groups(&a, &spans, "Fictional confirmed person");
    assert_eq!(
        groups,
        std::collections::BTreeSet::from(["Speaker 1".into()])
    );
    let selected = enrollment::confirmed_clean_intervals(&a, &spans, &[], &groups);
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().all(|i| i.speaker == "Speaker 1"));
    assert_eq!((selected[0].start, selected[0].end), (0., 4.));
    assert_eq!((selected[1].start, selected[1].end), (8., 12.));
}

#[test]
fn automatic_target_resolves_per_engine_without_changing_native_gpu_choice() {
    let target = config::ProcessingTarget::default();
    assert_eq!(target.device, "auto");
    let probe = serde_json::json!({"engines":[{"id":"whisper_cpp","model":"large-v3-turbo-q5_0","available":true,"verified":true,"device":"vulkan"},{"id":"faster_whisper","model":"large-v3-turbo-fp16","available":true,"verified":true,"device":"cpu"}]});
    let native = config::PipelineConfig::default();
    assert_eq!(
        service::verified_speech_device(&probe, &native).unwrap(),
        "vulkan"
    );
    let mut optional = native.clone();
    optional.speech_engine = config::SpeechEngine::FasterWhisper;
    optional.speech_model = "large-v3-turbo-fp16".into();
    assert_eq!(
        service::verified_speech_device(&probe, &optional).unwrap(),
        "cpu"
    );
    assert_eq!(
        service::verified_speech_device(&probe, &native).unwrap(),
        "vulkan"
    );
    assert_eq!(target.device, "auto");
    let unresolved = serde_json::json!({"engines":[{"id":"faster_whisper","model":"large-v3-turbo-fp16","available":true,"verified":true,"device":"auto"}]});
    assert!(service::verified_speech_device(&unresolved, &optional).is_err());
}

fn reference_voice(vector: Vec<f32>) -> types::SpeakerEmbedding {
    types::SpeakerEmbedding {
        speaker: "Speaker 1".into(),
        model_id: config::NATIVE_EMBEDDING.into(),
        vector,
        clean_seconds: 4.,
        sample_count: 1,
        embedding_space: Some(types::EmbeddingSpace {
            id: speaker_core::campplus_space_id("fictional-checkpoint-hash"),
            model: "campplus".into(),
            revision: "fictional-checkpoint-hash".into(),
            preprocessing: "pcm16k-mono-ffmpeg-v1".into(),
            normalization: "l2".into(),
            aggregation: "longest-three-trimmed-10s-v1".into(),
            extraction_version: "parley-native-v2".into(),
            dimension: 2,
        }),
        source_intervals: vec![types::SourceInterval {
            speaker: "Speaker 1".into(),
            start: 1.,
            end: 5.,
            channel: 0,
        }],
    }
}
async fn enrolled_fixture(p: &SqlitePool, m: &str, name: &str, vector: Vec<f32>) -> String {
    let id = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        p,
        name,
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &id, m, None, None, &reference_voice(vector))
        .await
        .unwrap();
    tx.commit().await.unwrap();
    id
}
fn identification_options(replace_confirmed: bool) -> service::IdentificationOptions {
    service::IdentificationOptions {
        identification: config::Identification::Campplus,
        target_id: "missing-target-must-not-be-needed-for-cached-evidence".into(),
        replace_confirmed,
    }
}

#[tokio::test]
async fn high_score_is_only_a_proposal_until_explicit_acceptance() {
    let p = pool().await;
    let mut a = analysis();
    a.embeddings.push(reference_voice(vec![1., 0.]));
    let m = meeting(&p, &a).await;
    let profile = enrolled_fixture(&p, &m, "Fictional Alice", vec![1., 0.]).await;
    let before: Vec<(String, Option<String>, Option<String>, Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT transcript,speaker,voice_profile_id,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=?"
    ).bind(&m).fetch_all(&p).await.unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let result =
        service::identify_speakers(&p, &m, identification_options(false), &cancel, |_, _, _| {
            panic!("cached identity evidence must not start inference")
        })
        .await
        .unwrap();
    assert_eq!(
        result.suggestions[0].profile_id.as_deref(),
        Some(profile.as_str())
    );
    assert!(!result.suggestions[0].confirmed);
    assert!(!result.suggestions[0].replacement);
    let after: Vec<(String, Option<String>, Option<String>, Option<f64>, Option<f64>)> = sqlx::query_as(
        "SELECT transcript,speaker,voice_profile_id,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=?"
    ).bind(&m).fetch_all(&p).await.unwrap();
    assert_eq!(
        before, after,
        "machine proposals cannot rewrite the transcript"
    );
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &profile)
        .await
        .unwrap();
    let (name, id): (String, String) =
        sqlx::query_as("SELECT speaker,voice_profile_id FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!((name, id), ("Fictional Alice".into(), profile.clone()));
    assert!(service::current_identity_suggestions(&p, &m).await.unwrap()[0].confirmed);
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(count, 1, "acceptance must not duplicate the same source audio");
    let policy: String =
        sqlx::query_scalar("SELECT policy_version FROM identification_runs WHERE id=?")
            .bind(&result.run_id)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(policy, "conservative-review-required-v4");
    let result: String =
        sqlx::query_scalar("SELECT result_json FROM identification_runs WHERE id=?")
            .bind(&result.run_id)
            .fetch_one(&p)
            .await
            .unwrap();
    let result: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(result["resolved_policy"]["context_status"], "resolved");
    assert_eq!(result["resolved_policy"]["campplus_baseline"], 0.75);
    assert!(result["resolved_policy"]["ambiguity_margin"]
        .as_f64()
        .is_some_and(|margin| (margin - 0.10).abs() < 0.000001));
    assert!(!result["resolved_policy"]["candidate_contexts"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(runs::redact_metadata(&result)
        .get("resolved_policy")
        .is_none());
}

#[tokio::test]
async fn alternatives_keep_confirmed_names_and_reject_stale_replacements() {
    let p = pool().await;
    let mut a = analysis();
    a.embeddings.push(reference_voice(vec![1., 0.]));
    let m = meeting(&p, &a).await;
    let alice = enrolled_fixture(&p, &m, "Fictional Alice", vec![0., 1.]).await;
    let bob = enrolled_fixture(&p, &m, "Fictional Bob", vec![1., 0.]).await;
    let carol = enrolled_fixture(&p, &m, "Fictional Carol", vec![0., -1.]).await;
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &alice)
        .await
        .unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let preserved =
        service::identify_speakers(&p, &m, identification_options(false), &cancel, |_, _, _| {})
            .await
            .unwrap();
    assert!(preserved.suggestions[0].confirmed);
    assert_eq!(
        preserved.suggestions[0].profile_id.as_deref(),
        Some(alice.as_str())
    );
    let proposed =
        service::identify_speakers(&p, &m, identification_options(true), &cancel, |_, _, _| {})
            .await
            .unwrap();
    let proposal = &proposed.suggestions[0];
    assert!(proposal.replacement);
    assert!(!proposal.confirmed);
    assert_eq!(proposal.profile_id.as_deref(), Some(bob.as_str()));
    assert_eq!(proposal.current_profile_id.as_deref(), Some(alice.as_str()));
    let current: String =
        sqlx::query_scalar("SELECT voice_profile_id FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(current, alice);
    assert!(service::current_identity_suggestions(&p, &m).await.unwrap()[0].replacement);
    assert!(
        service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &bob)
            .await
            .is_err()
    );
    let expected_alice = service::IdentityAssignment {
        profile_id: Some(alice.clone()),
        name: Some("Fictional Alice".into()),
    };
    assert!(service::confirm_identity_replacement(
        &p,
        &m,
        "stale-group",
        "Speaker 1",
        &bob,
        expected_alice.clone()
    )
    .await
    .is_err());
    service::confirm_identity_replacement(
        &p,
        &m,
        "example-job",
        "Speaker 1",
        &carol,
        expected_alice.clone(),
    )
    .await
    .unwrap();
    assert!(service::confirm_identity_replacement(
        &p,
        &m,
        "example-job",
        "Speaker 1",
        &bob,
        expected_alice
    )
    .await
    .is_err());
    let stale = service::current_identity_suggestions(&p, &m).await.unwrap();
    assert!(stale[0].confirmed);
    assert_eq!(stale[0].profile_id.as_deref(), Some(carol.as_str()));
    service::confirm_identity_replacement(
        &p,
        &m,
        "example-job",
        "Speaker 1",
        &bob,
        service::IdentityAssignment {
            profile_id: Some(carol),
            name: Some("Fictional Carol".into()),
        },
    )
    .await
    .unwrap();
    let accepted = service::current_identity_suggestions(&p, &m).await.unwrap();
    assert!(accepted[0].confirmed);
    assert!(!accepted[0].replacement);
    assert_eq!(accepted[0].profile_id.as_deref(), Some(bob.as_str()));
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(count, 1, "replaced identities must lose evidence from the corrected group");
}

#[tokio::test]
async fn failed_identification_records_the_same_review_policy() {
    let p = pool().await;
    let m = meeting(&p, &analysis()).await;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    assert!(service::identify_speakers(
        &p,
        &m,
        identification_options(false),
        &cancel,
        |_, _, _| {}
    )
    .await
    .is_err());
    let (policy, result): (String, String) = sqlx::query_as(
        "SELECT policy_version,result_json FROM identification_runs WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(policy, "conservative-review-required-v4");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&result).unwrap()["outcome"],
        "failed"
    );
}

#[test]
fn preparation_routes_artifacts_to_the_registry_each_adapter_reads() {
    use config::{Identification::*, SpeakerPipeline::*, SpeechEngine::*};
    let local = config::ProcessingTarget::default();
    let mut config = config::PipelineConfig::default();
    let p = preparation::plan(&local, &config, true);
    assert_eq!(p.app_speech.as_deref(), Some(config.speech_model.as_str()));
    assert!(p.app_campplus && p.app_segmentation);
    assert!(
        p.worker.is_empty(),
        "native preparation must work without Python"
    );
    let p = preparation::plan(&local, &config, false);
    assert!(p.app_speech.is_none());
    assert!(
        p.worker.is_empty(),
        "speaker-only preparation must not download ASR"
    );
    config.speech_engine = FasterWhisper;
    config.speech_model = "large-v3-turbo-int8".into();
    let p = preparation::plan(&local, &config, true);
    assert!(p.app_speech.is_none());
    assert!(p.app_campplus && p.app_segmentation);
    assert_eq!(p.worker, vec!["large-v3-turbo-int8"]);
    config.speech_engine = WhisperCpp;
    config.speaker_pipeline = Community1;
    config.identification = Campplus;
    let p = preparation::plan(&local, &config, true);
    assert!(p.app_speech.is_some() && p.app_campplus && !p.app_segmentation);
    assert_eq!(p.worker, vec!["community1"]);
    config.speaker_pipeline = Sherpa;
    config.identification = Wespeaker;
    let p = preparation::plan(&local, &config, false);
    assert!(p.app_speech.is_none() && p.app_campplus && p.app_segmentation);
    assert_eq!(p.worker, vec!["community1"]);
    let mut remote = local;
    remote.kind = config::TargetKind::Ssh;
    let p = preparation::plan(&remote, &config, false);
    assert!(!p.app_campplus && !p.app_segmentation && p.app_speech.is_none());
    assert_eq!(
        p.worker,
        vec!["campplus", "community1", "sherpa-segmentation"]
    );
    config.speaker_pipeline = Community1;
    config.identification = Same;
    let p = preparation::plan(&remote, &config, true);
    assert_eq!(p.worker, vec!["community1", "large-v3-turbo-int8"]);
}

#[test]
fn linked_sherpa_recovery_requires_supported_speech_only_job_and_completes_once() {
    assert!(service::require_speech_only_operation(
        &serde_json::json!({"protocol_version":1,"operations":["process","speaker_analysis"]})
    )
    .is_err());
    assert!(service::require_speech_only_operation(
        &serde_json::json!({"protocol_version":1,"operations":["speech_recognition"]})
    )
    .is_ok());
    let target = config::ProcessingTarget::default();
    let mut config = config::PipelineConfig::default();
    config.speech_engine = config::SpeechEngine::FasterWhisper;
    let mut a = analysis();
    a.settings["worker_operation"] = "speech_recognition".into();
    assert!(service::needs_local_sherpa_completion(&target, &config, &a));
    a.settings["local_sherpa_completed"] = true.into();
    assert!(!service::needs_local_sherpa_completion(
        &target, &config, &a
    ));
    a.settings["local_sherpa_completed"] = false.into();
    config.speaker_pipeline = config::SpeakerPipeline::Community1;
    assert!(!service::needs_local_sherpa_completion(
        &target, &config, &a
    ));
    config.speaker_pipeline = config::SpeakerPipeline::Sherpa;
    let mut remote = target;
    remote.kind = config::TargetKind::Ssh;
    assert!(!service::needs_local_sherpa_completion(
        &remote, &config, &a
    ));
}

#[test]
fn partial_word_alignment_preserves_all_asr_text_during_linked_speaker_annotation() {
    let mut a = analysis();
    a.provenance["word_alignment"] = serde_json::json!({"unaligned_words":1});
    a.words = serde_json::from_value(serde_json::json!([
        {"start":1.,"end":2.,"text":"Fictional","speaker":null},
        {"start":3.,"end":5.,"text":" example.","speaker":null}
    ]))
    .unwrap();
    a.segments[0].text = "Fictional complete example.".into();
    native::annotate(&mut a);
    assert_eq!(a.segments.len(), 1);
    assert_eq!(a.segments[0].text, "Fictional complete example.");
    assert_eq!((a.segments[0].start, a.segments[0].end), (1., 5.));
    assert_eq!(a.segments[0].speaker.as_deref(), Some("Speaker 1"));
    assert!(a
        .words
        .iter()
        .all(|w| w.speaker.as_deref() == Some("Speaker 1")));
}

#[tokio::test]
async fn request_history_separates_requested_and_detected_language_and_retains_auto() {
    let p = pool().await;
    let target = config::ProcessingTarget::default();
    let options = service::ProcessingOptions {
        config: config::PipelineConfig::default(),
        language: Some("es".into()),
        num_speakers: 3,
    };
    let mut a = analysis();
    let started = "2026-01-02T03:04:05Z";
    service::snapshot_request(
        &mut a,
        &options,
        "explicit-language-request",
        "process",
        started,
        "fictional-hash",
        &target,
    )
    .unwrap();
    assert_eq!(
        a.language.as_deref(),
        Some("en"),
        "requested language cannot rewrite the detected result"
    );
    let m = meeting(&p, &a).await;
    let run = runs::list_runs(&p, &m).await.unwrap().remove(0);
    assert_eq!(run.started_at, started);
    assert_eq!(run.input["requested_language"], "es");
    assert_eq!(run.input["detected_language"], "en");
    assert_eq!(run.input["expected_speaker_count"], 3);
    assert_eq!(run.input["request_snapshot_available"], true);
    let auto = service::ProcessingOptions {
        language: None,
        num_speakers: 0,
        ..options
    };
    service::snapshot_request(
        &mut a,
        &auto,
        "auto-language-request",
        "process",
        started,
        "fictional-hash",
        &target,
    )
    .unwrap();
    assert!(
        a.settings
            .get("requested_language")
            .is_some_and(|v| v.is_null()),
        "explicit null means requested automatic detection"
    );
    let mut tx = p.begin().await.unwrap();
    runs::store_tx(&mut tx, &m, Some(&a)).await.unwrap();
    tx.commit().await.unwrap();
    let history = runs::list_runs(&p, &m).await.unwrap();
    let run = history
        .iter()
        .find(|r| r.id == "auto-language-request")
        .unwrap();
    assert!(run.input["requested_language"].is_null());
    assert_eq!(run.input["detected_language"], "en");
    assert_eq!(run.input["expected_speaker_count"], 0);
    assert_eq!(run.input["request_snapshot_available"], true);
}

#[tokio::test]
async fn recovery_keeps_original_target_and_options_after_target_edit_or_deletion() {
    let p = pool().await;
    let m = meeting(&p, &analysis()).await;
    let target = config::ProcessingTarget {
        id: "fictional-remote".into(),
        name: "Original target".into(),
        kind: config::TargetKind::Ssh,
        ssh_host: Some("fictional-original-alias".into()),
        worker_path: Some("fictional-original-worker".into()),
        python: Some("fictional-original-python".into()),
        device: "cuda".into(),
        ..Default::default()
    };
    config::save_target(&p, &target).await.unwrap();
    let options = service::ProcessingOptions {
        config: config::PipelineConfig {
            target_id: target.id.clone(),
            ..Default::default()
        },
        language: Some("es".into()),
        num_speakers: 4,
    };
    let request = "fictional-recoverable-job";
    let started = "2026-01-02T03:04:05Z";
    recovery::retain(&p, &m, request, &target, &options, "process", started)
        .await
        .unwrap();
    let mut edited = target.clone();
    edited.ssh_host = Some("fictional-different-alias".into());
    edited.worker_path = Some("fictional-different-worker".into());
    config::save_target(&p, &edited).await.unwrap();
    assert!(
        recovery::retain(&p, &m, request, &edited, &options, "process", started)
            .await
            .is_err(),
        "a request must never replace its connection identity"
    );
    config::delete_target(&p, &target.id).await.unwrap();
    assert!(config::get_target(&p, &target.id).await.is_err());
    let original = recovery::load(&p, &m, request).await.unwrap();
    assert_eq!(original.target.ssh_host, target.ssh_host);
    assert_eq!(original.target.worker_path, target.worker_path);
    assert_eq!(original.target.python, target.python);
    assert_eq!(original.options.language.as_deref(), Some("es"));
    assert_eq!(original.options.num_speakers, 4);
    assert_eq!(original.started_at, started);
    assert!(recovery::load(&p, "different-meeting", request)
        .await
        .is_err());
    let mut recovered = analysis();
    service::snapshot_request(
        &mut recovered,
        &original.options,
        request,
        &original.operation,
        &original.started_at,
        "fictional-hash",
        &original.target,
    )
    .unwrap();
    let public = runs::redact_metadata(&serde_json::to_value(&recovered).unwrap()).to_string();
    assert!(!public.contains("fictional-original-alias"));
    assert!(!public.contains("fictional-original-worker"));
    assert!(!public.contains("fictional-original-python"));
    let mut tx = p.begin().await.unwrap();
    runs::store_tx(&mut tx, &m, Some(&recovered)).await.unwrap();
    tx.commit().await.unwrap();
    let history = runs::list_runs(&p, &m).await.unwrap();
    let run = history.iter().find(|r| r.id == request).unwrap();
    assert_eq!(run.started_at, started);
    assert_eq!(run.input["requested_language"], "es");
    assert_eq!(run.input["expected_speaker_count"], 4);
    recovery::forget(&p, request).await.unwrap();
    assert!(
        recovery::load(&p, &m, request).await.is_err(),
        "historical jobs without a snapshot must not silently use the current target"
    );
    recovery::retain(
        &p,
        &m,
        "fictional-delete-recovery",
        &target,
        &options,
        "process",
        started,
    )
    .await
    .unwrap();
    crate::database::repositories::meeting::MeetingsRepository::delete_meeting(&p, &m)
        .await
        .unwrap();
    let remaining: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM app_settings WHERE key LIKE 'processing_recovery_v1:%'",
    )
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(
        remaining, 0,
        "deleting the meeting removes its private transport snapshots"
    );
}

#[tokio::test]
async fn default_local_choices_include_optional_packages_without_models_or_configuration_edits() {
    let p = pool().await;
    let choices = service::target_choices(&p, "local").await.unwrap();
    assert!(choices.engines.contains(&config::SpeechEngine::WhisperCpp));
    assert!(choices
        .engines
        .contains(&config::SpeechEngine::FasterWhisper));
    assert!(choices
        .speaker_pipelines
        .contains(&config::SpeakerPipeline::Sherpa));
    assert!(choices
        .speaker_pipelines
        .contains(&config::SpeakerPipeline::Community1));
    let settings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM app_settings")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(
        settings, 0,
        "listing installable choices neither rewrites defaults nor installs a runtime"
    );
}

#[test]
fn selectable_options_follow_architecture_instead_of_installed_or_verified_models() {
    let linux = config::choices_for_platform(
        &serde_json::json!({"system":"Linux","architecture":"x86_64"}),
    );
    assert_eq!(
        linux.engines,
        vec![
            config::SpeechEngine::WhisperCpp,
            config::SpeechEngine::FasterWhisper
        ]
    );
    assert_eq!(
        linux.speaker_pipelines,
        vec![
            config::SpeakerPipeline::Sherpa,
            config::SpeakerPipeline::Community1
        ]
    );
    let apple = config::choices_for_platform(
        &serde_json::json!({"system":"Darwin","architecture":"arm64"}),
    );
    assert!(apple.engines.contains(&config::SpeechEngine::MlxWhisper));
    let intel_apple = config::choices_for_platform(
        &serde_json::json!({"system":"Darwin","architecture":"x86_64"}),
    );
    assert!(!intel_apple
        .engines
        .contains(&config::SpeechEngine::MlxWhisper));
    assert!(config::choices_for_platform(
        &serde_json::json!({"system":"Linux","architecture":"unsupported-fixture"})
    )
    .engines
    .is_empty());
    assert!(
        config::choices_for_platform(&serde_json::json!({}))
            .engines
            .is_empty(),
        "unknown remote architecture must not inherit the controlling computer's options"
    );
}

#[test]
fn both_cross_model_identification_dependencies_gate_the_final_pipeline() {
    use config::{Identification::*, SpeakerPipeline::*};
    for (pipeline, selected, expected_extra) in [
        (Sherpa, Wespeaker, Some(Wespeaker)),
        (Community1, Campplus, Some(Campplus)),
        (Sherpa, Campplus, None),
        (Community1, Wespeaker, None),
        (Sherpa, Same, None),
        (Community1, Same, None),
        (Sherpa, Off, None),
        (Community1, Off, None),
    ] {
        let c = config::PipelineConfig {
            speaker_pipeline: pipeline,
            identification: selected,
            ..Default::default()
        };
        assert_eq!(service::additional_identification(&c), expected_extra);
        if let Some(extra) = expected_extra {
            let mut report = config::Readiness {
                ready: true,
                capabilities: serde_json::json!({"selected_speech_and_speakers_verified":true}),
                ..Default::default()
            };
            let absent = config::Readiness {
                ready: false,
                issues: vec!["Install or download the selected extractor".into()],
                ..Default::default()
            };
            service::include_identification_readiness(&mut report, extra, &absent);
            assert!(!report.ready);
            assert!(report
                .issues
                .iter()
                .any(|issue| issue.contains("Independent identification")));
            let mut complete = config::Readiness {
                ready: true,
                capabilities: serde_json::json!({"speaker_device":"cuda"}),
                ..Default::default()
            };
            let available = config::Readiness {
                ready: true,
                speaker_device: Some("cpu".into()),
                ..Default::default()
            };
            service::include_identification_readiness(&mut complete, extra, &available);
            assert!(complete.ready && complete.issues.is_empty());
            assert_eq!(
                complete.capabilities["identification"]["device"], "cpu",
                "the independent extractor retains its own resolved device"
            );
        }
    }
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn saved_mlx_choice_on_linux_requires_a_compatible_engine_before_model_probes() {
    let p = pool().await;
    let config = config::PipelineConfig {
        speech_engine: config::SpeechEngine::MlxWhisper,
        speech_model: "large-v3-turbo-mlx-fp16".into(),
        ..Default::default()
    };
    let report = service::check_readiness(&p, &config).await.unwrap();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = events.clone();
    let with_progress =
        service::check_readiness_with_progress(&p, &config, move |message, percent| {
            recorded.lock().unwrap().push((message.to_owned(), percent));
        })
        .await
        .unwrap();
    assert_eq!(
        serde_json::to_value(&report).unwrap(),
        serde_json::to_value(&with_progress).unwrap(),
        "the legacy wrapper preserves the readiness result"
    );
    assert_eq!(
        *events.lock().unwrap(),
        vec![("Checking target connection…".to_owned(), None)],
        "unsupported engines return before any model check or integrity progress"
    );
    assert!(!report.ready);
    assert!(!report.engines.contains(&config::SpeechEngine::MlxWhisper));
    assert!(report
        .engines
        .contains(&config::SpeechEngine::FasterWhisper));
    assert_eq!(
        report.issues,
        vec![
        "Choose a speech engine compatible with this target's operating system and architecture"
    ]
    );
    assert!(
        report.capabilities.is_null(),
        "unsupported selections return before model/runtime probing"
    );
    let settings: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM app_settings")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(settings, 0);
}

#[tokio::test]
async fn confirmation_committed_after_result_preparation_survives_atomic_publication() {
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    // The background job prepared anonymous rows before a human confirmed them.
    a.settings["request_id"] = "result-prepared-before-confirmation".into();
    a.segments[0].text = "Fictional updated words.".into();
    let prepared = a.transcript_segments(None).await.unwrap();
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional confirmed Alice",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &profile)
        .await
        .unwrap();
    // Publication must reconcile against the current transaction, not the
    // anonymous labels captured while the job was running.
    crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis_cancel(
        &p, &m, &prepared, Some(&a), Some(&std::sync::atomic::AtomicBool::new(false)),
    ).await.unwrap();
    let row: (String, String, String) = sqlx::query_as(
        "SELECT transcript,speaker,voice_profile_id FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(
        row,
        (
            "Fictional updated words.".into(),
            "Fictional confirmed Alice".into(),
            profile.clone()
        )
    );
    assert_eq!(
        runs::current_run(&p, &m).await.unwrap().as_deref(),
        Some("result-prepared-before-confirmation")
    );
    let current = service::current_identity_suggestions(&p, &m).await.unwrap();
    assert!(current[0].confirmed);
    assert_eq!(current[0].profile_id.as_deref(), Some(profile.as_str()));
    assert!(enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn speaker_only_publication_reads_human_changes_at_its_transaction_boundary() {
    let p = pool().await;
    let mut new_groups = analysis();
    let m = meeting(&p, &new_groups).await;
    // A completed detector result has new anonymous group keys. It must not
    // reuse the transcript's labels from before the human's confirmation.
    new_groups.settings["request_id"] = "groups-prepared-before-confirmation".into();
    new_groups.settings["operation"] = "speaker_analysis".into();
    new_groups.turns[0].speaker = "Speaker 2".into();
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional confirmed Bob",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &profile)
        .await
        .unwrap();
    let before: (String,String,String,String,f64,f64) = sqlx::query_as("SELECT id,transcript,speaker,voice_profile_id,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=?")
        .bind(&m).fetch_one(&p).await.unwrap();
    service::publish_speaker_analysis(
        &p,
        &m,
        new_groups,
        &std::sync::atomic::AtomicBool::new(false),
    )
    .await
    .unwrap();
    let after: (String,String,String,String,f64,f64) = sqlx::query_as("SELECT id,transcript,speaker,voice_profile_id,audio_start_time,audio_end_time FROM transcripts WHERE meeting_id=?")
        .bind(&m).fetch_one(&p).await.unwrap();
    assert_eq!(
        before, after,
        "re-detect keeps current human identity and every transcript row boundary"
    );
    let analysis = runs::load(&p, &m).await.unwrap().unwrap();
    assert_eq!(analysis.turns[0].speaker, "Speaker 2");
    let current = service::current_identity_suggestions(&p, &m).await.unwrap();
    assert_eq!(current[0].speaker, "Speaker 2");
    assert!(current[0].confirmed);
    assert_eq!(current[0].profile_id.as_deref(), Some(profile.as_str()));
}

#[tokio::test]
async fn unavailable_legacy_human_timing_aborts_replacement_without_losing_words_or_history() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    sqlx::query("UPDATE transcripts SET speaker='Fictional human name',identity_origin='legacy_unknown',audio_start_time=NULL,audio_end_time=NULL WHERE meeting_id=?")
        .bind(&m).execute(&p).await.unwrap();
    let mut replacement = a.clone();
    replacement.settings["request_id"] = "cannot-reconcile-untimed-human".into();
    replacement.segments[0].text = "Fictional replacement.".into();
    let rows = replacement.transcript_segments(None).await.unwrap();
    let error = crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis(
        &p, &m, &rows, Some(&replacement),
    ).await.unwrap_err();
    assert!(error
        .to_string()
        .contains("human speaker assignment has no audio timing"));
    let row: (String, String) =
        sqlx::query_as("SELECT transcript,speaker FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        row,
        ("Fictional example.".into(), "Fictional human name".into())
    );
    assert_eq!(
        runs::current_run(&p, &m).await.unwrap().as_deref(),
        Some("example-job")
    );
    assert_eq!(runs::list_runs(&p, &m).await.unwrap().len(), 1);
}

#[tokio::test]
async fn equal_time_publication_keeps_sequence_confidence_and_each_original_row_field() {
    use sqlx::Row;
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    a.settings["request_id"] = "equal-time-canonical-order".into();
    a.segments[0].text = "Fictional first voice.".into();
    let mut overlapping = a.segments[0].clone();
    overlapping.text = "Fictional simultaneous voice.".into();
    a.segments.push(overlapping);
    let mut rows = a.transcript_segments(None).await.unwrap();
    rows[0].confidence = Some(0.75);
    rows[0].source = Some("mic".into());
    rows[1].confidence = Some(0.875);
    rows[1].source = Some("system".into());
    crate::database::repositories::transcript::TranscriptsRepository::replace_transcripts_with_analysis(
        &p, &m, &rows, Some(&a),
    ).await.unwrap();
    let persisted = sqlx::query("SELECT id,transcript,audio_start_time,audio_end_time,duration,speaker,voice_profile_id,sequence_id,source,confidence FROM transcripts WHERE meeting_id=? ORDER BY sequence_id")
        .bind(&m).fetch_all(&p).await.unwrap();
    assert_eq!(persisted.len(), 2);
    for (actual, expected) in persisted.iter().zip(&rows) {
        assert_eq!(actual.get::<String, _>("id"), expected.id);
        assert_eq!(actual.get::<String, _>("transcript"), expected.text);
        assert_eq!(
            actual.get::<Option<f64>, _>("audio_start_time"),
            expected.audio_start_time
        );
        assert_eq!(
            actual.get::<Option<f64>, _>("audio_end_time"),
            expected.audio_end_time
        );
        assert_eq!(actual.get::<Option<f64>, _>("duration"), expected.duration);
        assert_eq!(actual.get::<Option<String>, _>("speaker"), expected.speaker);
        assert_eq!(
            actual.get::<Option<String>, _>("voice_profile_id"),
            expected.voice_profile_id
        );
        assert_eq!(
            actual.get::<Option<i64>, _>("sequence_id"),
            expected.sequence_id.map(|sequence| sequence as i64)
        );
        assert_eq!(actual.get::<Option<String>, _>("source"), expected.source);
        assert_eq!(
            actual.get::<Option<f64>, _>("confidence"),
            expected.confidence.map(|confidence| confidence as f64)
        );
    }
    assert_eq!(
        runs::current_run(&p, &m).await.unwrap().as_deref(),
        Some("equal-time-canonical-order")
    );
}

#[tokio::test]
async fn merging_profiles_combines_only_exact_spaces_and_preserves_every_evidence_source() {
    use crate::database::repositories::voice_profile::VoiceProfilesRepository;
    let p = pool().await;
    let mut a = analysis();
    a.settings["request_id"] = "fictional-merge-source-a".into();
    let meeting_a = meeting(&p, &a).await;
    a.settings["request_id"] = "fictional-merge-source-b".into();
    let meeting_b = meeting(&p, &a).await;
    let winner = VoiceProfilesRepository::create(&p, "Fictional retained identity", None, &[], 0)
        .await
        .unwrap();
    let loser = VoiceProfilesRepository::create(&p, "Fictional duplicate identity", None, &[], 0)
        .await
        .unwrap();
    service::confirm_identity_suggestion(
        &p,
        &meeting_a,
        "fictional-merge-source-a",
        "Speaker 1",
        &winner,
    )
    .await
    .unwrap();
    service::confirm_identity_suggestion(
        &p,
        &meeting_b,
        "fictional-merge-source-b",
        "Speaker 1",
        &loser,
    )
    .await
    .unwrap();

    let winner_voice = reference_voice(vec![1., 0.]);
    let mut loser_same_space = reference_voice(vec![0., 1.]);
    loser_same_space.source_intervals[0].start = 2.;
    loser_same_space.source_intervals[0].end = 6.;
    let mut loser_other_space = reference_voice(vec![-1., 0.]);
    let other = loser_other_space.embedding_space.as_mut().unwrap();
    other.id = speaker_core::campplus_space_id("fictional-other-checkpoint-hash");
    other.revision = "fictional-other-checkpoint-hash".into();
    loser_other_space.source_intervals[0].start = 6.;
    loser_other_space.source_intervals[0].end = 10.;
    let shared_space = winner_voice.embedding_space.as_ref().unwrap().id.clone();
    let other_space = loser_other_space
        .embedding_space
        .as_ref()
        .unwrap()
        .id
        .clone();
    let hash_a = "a".repeat(64);
    let hash_b = "b".repeat(64);
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(
        &mut tx,
        &winner,
        &meeting_a,
        Some("fictional-merge-source-a"),
        Some(&hash_a),
        &winner_voice,
    )
    .await
    .unwrap();
    for voice in [&loser_same_space, &loser_other_space] {
        enrollment::store_evidence_tx(
            &mut tx,
            &loser,
            &meeting_b,
            Some("fictional-merge-source-b"),
            Some(&hash_b),
            voice,
        )
        .await
        .unwrap();
    }
    tx.commit().await.unwrap();

    // Exclude only the intentionally changed owner. Every byte of sample
    // identity, vector, source window and human-confirmation provenance survives.
    type Evidence = (
        String,
        String,
        Option<String>,
        Option<String>,
        i64,
        String,
        Option<String>,
        String,
        Option<String>,
        f64,
        String,
    );
    let evidence_sql = "SELECT id,space_id,source_meeting_id,audio_sha256,channel,intervals_json,source_run_id,confirmation_json,embedding_json,clean_seconds,created_at FROM enrollment_samples ORDER BY id";
    let before: Vec<Evidence> = sqlx::query_as(evidence_sql).fetch_all(&p).await.unwrap();
    assert_eq!(before.len(), 3);
    let legacy_revisions: Vec<(String,String,String,String)> = sqlx::query_as("SELECT profile_id,space_id,revision,summary_json FROM enrollment_versions ORDER BY profile_id,space_id,revision").fetch_all(&p).await.unwrap();
    let outcome = VoiceProfilesRepository::merge_into(&p, &winner, &loser)
        .await
        .unwrap();
    assert_eq!(outcome.relinked_transcripts, 1);
    assert!(outcome.centroid_updated);
    assert!(VoiceProfilesRepository::get_by_id(&p, &loser)
        .await
        .unwrap()
        .is_none());
    let after: Vec<Evidence> = sqlx::query_as(evidence_sql).fetch_all(&p).await.unwrap();
    assert_eq!(before, after);
    let owners: Vec<String> =
        sqlx::query_scalar("SELECT DISTINCT profile_id FROM enrollment_samples")
            .fetch_all(&p)
            .await
            .unwrap();
    assert_eq!(owners, vec![winner.clone()]);
    let enrollments = enrollment::list_enrollments(&p, &winner).await.unwrap();
    assert_eq!(enrollments.len(), 2);
    let shared = enrollments
        .iter()
        .find(|e| e.space_id == shared_space)
        .unwrap();
    let separate = enrollments
        .iter()
        .find(|e| e.space_id == other_space)
        .unwrap();
    assert_eq!(
        (
            shared.status.as_str(),
            shared.sample_count,
            shared.clean_seconds
        ),
        ("ready", 2, 8.)
    );
    assert_eq!(
        (
            separate.status.as_str(),
            separate.sample_count,
            separate.clean_seconds
        ),
        ("ready", 1, 4.)
    );
    assert_eq!(
        shared.embedding_space.as_ref().unwrap().revision,
        "fictional-checkpoint-hash"
    );
    assert_eq!(
        separate.embedding_space.as_ref().unwrap().revision,
        "fictional-other-checkpoint-hash"
    );
    let centroid = |json: String| serde_json::from_str::<Vec<f32>>(&json).unwrap();
    let shared_json: String = sqlx::query_scalar(
        "SELECT embedding_json FROM profile_enrollments WHERE profile_id=? AND space_id=?",
    )
    .bind(&winner)
    .bind(&shared_space)
    .fetch_one(&p)
    .await
    .unwrap();
    let combined = centroid(shared_json);
    assert!((combined[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.000001);
    assert!((combined[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.000001);
    let other_json: String = sqlx::query_scalar(
        "SELECT embedding_json FROM profile_enrollments WHERE profile_id=? AND space_id=?",
    )
    .bind(&winner)
    .bind(&other_space)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(
        centroid(other_json),
        vec![-1., 0.],
        "equal dimensions and model family cannot cross a checkpoint boundary"
    );
    let linked: Vec<(String, String)> =
        sqlx::query_as("SELECT speaker,voice_profile_id FROM transcripts ORDER BY meeting_id")
            .fetch_all(&p)
            .await
            .unwrap();
    assert_eq!(linked.len(), 2);
    assert!(linked
        .iter()
        .all(|(name, profile)| name == "Fictional retained identity" && profile == &winner));
    let decisions: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT profile_id FROM speaker_identity_decisions WHERE confirmed=1",
    )
    .fetch_all(&p)
    .await
    .unwrap();
    assert_eq!(decisions, vec![winner.clone()]);
    assert!(enrollment::list_enrollments(&p, &loser)
        .await
        .unwrap()
        .is_empty());
    for old in legacy_revisions {
        let preserved: (String,String,String,String) = sqlx::query_as("SELECT profile_id,space_id,revision,summary_json FROM enrollment_versions WHERE profile_id=? AND space_id=? AND revision=?")
            .bind(&old.0).bind(&old.1).bind(&old.2).fetch_one(&p).await.unwrap();
        assert_eq!(
            old, preserved,
            "immutable enrollment history retains its original provenance"
        );
    }
}

#[test]
fn heuristic_ambiguity_checks_every_finite_competitor_without_threshold_tuning() {
    use crate::speaker_diarization::matching_policy as policy;
    // The third raw competitor ranks behind an invited runner-up after the
    // attendee penalty. It must still prevent a misleading proposal.
    assert_eq!(
        policy::saved_choice(&[(0.95, 0.75), (0.84, 0.75), (0.98, 0.90)]),
        None
    );
    // A competitor below its own threshold is still ambiguous evidence.
    assert_eq!(policy::saved_choice(&[(0.95, 0.75), (0.89, 0.90)]), None);
    assert_eq!(
        policy::saved_choice(&[(0.95, 0.75), (0.80, 0.90), (f32::NAN, 0.75)]),
        Some(0)
    );
    assert_eq!(policy::saved_baseline("campplus"), 0.75);
    assert_eq!(policy::saved_baseline("wespeaker"), 0.60);
    assert_eq!(policy::AMBIGUITY_MARGIN, 0.10);
}

#[tokio::test]
async fn semantic_enrollment_retry_is_idempotent_and_keeps_distinct_sources_spaces_and_intervals() {
    let p = pool().await;
    let a = analysis();
    let m = meeting(&p, &a).await;
    let mut copied = a.clone();
    copied.settings["request_id"] = "fictional-copied-source".into();
    let other_meeting = meeting(&p, &copied).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional evidence retry",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let mut e = reference_voice(vec![1., 0.]);
    e.source_intervals.push(types::SourceInterval {
        speaker: e.speaker.clone(),
        start: 6.,
        end: 10.,
        channel: 0,
    });
    e.clean_seconds = 8.;
    e.sample_count = 2;
    let hash = "c".repeat(64);
    let space = e.embedding_space.as_ref().unwrap().id.clone();
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, Some("original-run"), Some(&hash), &e)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let before:(String,String,f64,String)=sqlx::query_as("SELECT revision,embedding_json,clean_seconds,updated_at FROM profile_enrollments WHERE profile_id=?").bind(&profile).fetch_one(&p).await.unwrap();
    let versions: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_versions")
        .fetch_one(&p)
        .await
        .unwrap();
    let mut retry = e.clone();
    retry.speaker = "Renumbered group".into();
    retry.source_intervals.reverse();
    for interval in &mut retry.source_intervals {
        interval.speaker = retry.speaker.clone();
    }
    retry.vector[1] = 0.0000001;
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(
        &mut tx,
        &profile,
        &other_meeting,
        Some("different-run"),
        Some(&hash),
        &retry,
    )
    .await
    .unwrap();
    tx.commit().await.unwrap();
    let after:(String,String,f64,String)=sqlx::query_as("SELECT revision,embedding_json,clean_seconds,updated_at FROM profile_enrollments WHERE profile_id=?").bind(&profile).fetch_one(&p).await.unwrap();
    assert_eq!(before, after);
    assert_eq!(
        enrollment::list_enrollments(&p, &profile).await.unwrap()[0].sample_count,
        1
    );
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples")
        .fetch_one(&p)
        .await
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM enrollment_versions")
            .fetch_one(&p)
            .await
            .unwrap(),
        versions
    );
    let mut near = e.clone();
    near.source_intervals[0].start = f64::from_bits(1f64.to_bits() + 1);
    let mut different_space = e.clone();
    let metadata = different_space.embedding_space.as_mut().unwrap();
    metadata.id = speaker_core::campplus_space_id("different-retry-checkpoint");
    metadata.revision = "different-retry-checkpoint".into();
    let different_hash = "d".repeat(64);
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, Some(&hash), &near)
        .await
        .unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, Some(&different_hash), &e)
        .await
        .unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, Some(&hash), &different_space)
        .await
        .unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, None, &e)
        .await
        .unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &other_meeting, None, None, &e)
        .await
        .unwrap();
    // Missing recording identities must not turn unrelated legacy samples into
    // one contribution after their source meetings are removed.
    sqlx::query("UPDATE enrollment_samples SET source_meeting_id=NULL WHERE audio_sha256 IS NULL")
        .execute(&mut *tx)
        .await
        .unwrap();
    enrollment::rebuild_tx(&mut tx, &profile, &space)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let enrolled = enrollment::list_enrollments(&p, &profile).await.unwrap();
    assert_eq!(
        enrolled
            .iter()
            .find(|entry| entry.space_id == space)
            .unwrap()
            .sample_count,
        5
    );
    assert_eq!(
        sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM enrollment_samples")
            .fetch_one(&p)
            .await
            .unwrap(),
        6
    );
}

#[tokio::test]
async fn merged_duplicate_audio_counts_once_and_removing_any_alias_removes_the_contribution() {
    for remove_alias in [false, true] {
        let p = pool().await;
        let m = meeting(&p, &analysis()).await;
        let winner = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
            &p,
            "Fictional winner",
            None,
            &[],
            0,
        )
        .await
        .unwrap();
        let loser = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
            &p,
            "Fictional alias",
            None,
            &[],
            0,
        )
        .await
        .unwrap();
        let e = reference_voice(vec![1., 0.]);
        let space = e.embedding_space.as_ref().unwrap().id.clone();
        let mut distinct = reference_voice(vec![0., 1.]);
        distinct.source_intervals[0].start = 6.;
        distinct.source_intervals[0].end = 10.;
        let mut other = e.clone();
        let metadata = other.embedding_space.as_mut().unwrap();
        metadata.id = speaker_core::campplus_space_id("other-merge-space");
        metadata.revision = "other-merge-space".into();
        let other_space = metadata.id.clone();
        let hash = "e".repeat(64);
        let mut tx = p.begin().await.unwrap();
        enrollment::store_evidence_tx(&mut tx, &winner, &m, None, Some(&hash), &e)
            .await
            .unwrap();
        enrollment::store_evidence_tx(&mut tx, &loser, &m, None, Some(&hash), &e)
            .await
            .unwrap();
        enrollment::store_evidence_tx(&mut tx, &winner, &m, None, Some(&hash), &distinct)
            .await
            .unwrap();
        enrollment::store_evidence_tx(&mut tx, &winner, &m, None, Some(&hash), &other)
            .await
            .unwrap();
        tx.commit().await.unwrap();
        crate::database::repositories::voice_profile::VoiceProfilesRepository::merge_into(
            &p, &winner, &loser,
        )
        .await
        .unwrap();
        let entries = enrollment::list_enrollments(&p, &winner).await.unwrap();
        let main = entries
            .iter()
            .find(|entry| entry.space_id == space)
            .unwrap();
        assert_eq!((main.sample_count, main.clean_seconds), (2, 8.));
        let json: String = sqlx::query_scalar(
            "SELECT embedding_json FROM profile_enrollments WHERE profile_id=? AND space_id=?",
        )
        .bind(&winner)
        .bind(&space)
        .fetch_one(&p)
        .await
        .unwrap();
        let vector: Vec<f32> = serde_json::from_str(&json).unwrap();
        assert!((vector[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 0.000001);
        let samples = enrollment::list_samples(&p, &winner, &space).await.unwrap();
        assert_eq!(samples.len(), 3);
        let alias = samples
            .iter()
            .find(|sample| sample.duplicate_of.is_some())
            .unwrap();
        assert!(!alias.contributes);
        assert_eq!(alias.quality_status, "duplicate_reference");
        let selected = if remove_alias {
            alias.id.clone()
        } else {
            alias.duplicate_of.clone().unwrap()
        };
        let untouched:(String,String)=sqlx::query_as("SELECT revision,embedding_json FROM profile_enrollments WHERE profile_id=? AND space_id=?").bind(&winner).bind(&other_space).fetch_one(&p).await.unwrap();
        enrollment::remove_sample(&p, &selected).await.unwrap();
        let remaining = enrollment::list_samples(&p, &winner, &space).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert!(remaining[0].contributes);
        assert_eq!(remaining[0].intervals[0].start, 6.);
        let main = enrollment::list_enrollments(&p, &winner)
            .await
            .unwrap()
            .into_iter()
            .find(|entry| entry.space_id == space)
            .unwrap();
        assert_eq!((main.sample_count, main.clean_seconds), (1, 4.));
        let after:(String,String)=sqlx::query_as("SELECT revision,embedding_json FROM profile_enrollments WHERE profile_id=? AND space_id=?").bind(&winner).bind(&other_space).fetch_one(&p).await.unwrap();
        assert_eq!(untouched, after);
        assert_eq!(
            sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM enrollment_samples")
                .fetch_one(&p)
                .await
                .unwrap(),
            2
        );
    }
}

#[tokio::test]
async fn zero_norm_enrollment_centroid_is_insufficient_instead_of_ready() {
    let p = pool().await;
    let m = meeting(&p, &analysis()).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "Fictional conflicting evidence",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let e = reference_voice(vec![1., 0.]);
    let mut opposite = reference_voice(vec![-1., 0.]);
    opposite.source_intervals[0].start = 6.;
    opposite.source_intervals[0].end = 10.;
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, None, &e)
        .await
        .unwrap();
    enrollment::store_evidence_tx(&mut tx, &profile, &m, None, None, &opposite)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let entry = enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .remove(0);
    assert_eq!(entry.status, "insufficient_clean_evidence");
    let centroid: Option<String> =
        sqlx::query_scalar("SELECT embedding_json FROM profile_enrollments WHERE profile_id=?")
            .bind(&profile)
            .fetch_one(&p)
            .await
            .unwrap();
    assert!(centroid.is_none());
    assert!(identification::match_profiles_for_meeting(&p, &m, &[e])
        .await
        .unwrap()
        .is_empty());
}

#[tokio::test]
async fn saved_and_live_share_cautious_self_alias_context_with_distinct_baselines() {
    use crate::speaker_diarization::{matching_policy as policy, SpeakerProfileMatcher};
    let p = pool().await;
    let m = meeting(&p, &analysis()).await;
    let own = crate::database::repositories::voice_profile::VoiceProfilesRepository::upsert_self(
        &p,
        "Fictional Owner",
        &[],
        0,
    )
    .await
    .unwrap();
    let alias = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p,
        "  fictional OWNER ",
        None,
        &[],
        0,
    )
    .await
    .unwrap();
    let query = reference_voice(vec![1., 0.]);
    let mut tx = p.begin().await.unwrap();
    enrollment::store_evidence_tx(
        &mut tx,
        &own,
        &m,
        None,
        None,
        &reference_voice(vec![0., 1.]),
    )
    .await
    .unwrap();
    enrollment::store_evidence_tx(&mut tx, &alias, &m, None, None, &query)
        .await
        .unwrap();
    tx.commit().await.unwrap();
    let live = SpeakerProfileMatcher::new(
        2,
        vec![
            (own.clone(), "Fictional Owner".into(), vec![0., 1.]),
            (alias.clone(), "  fictional OWNER ".into(), vec![1., 0.]),
        ],
        0.60,
    )
    .unwrap()
    .with_self_profile(Some(own.clone()));
    live.set_expected_profiles(Some(std::collections::HashSet::new()));
    assert!(live.is_self(&alias));
    assert_eq!(live.threshold_for(&alias), 0.60);
    assert_eq!(policy::effective_threshold(0.75, true, Some(false)), 0.75);
    let report = identification::match_with_policy(&p, &m, &[query.clone()])
        .await
        .unwrap();
    let context = report.resolved_policy["candidate_contexts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["profile_id"] == alias)
        .unwrap();
    assert_eq!(context["self_or_same_name_alias"], true);
    assert_eq!(
        report.resolved_policy["version"],
        "conservative-review-required-v4"
    );
    enrollment::delete_enrollment(&p, &own, &query.embedding_space.as_ref().unwrap().id)
        .await
        .unwrap();
    let report = identification::match_with_policy(&p, &m, &[query])
        .await
        .unwrap();
    let context = report.resolved_policy["candidate_contexts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["profile_id"] == alias)
        .unwrap();
    assert_eq!(
        context["self_or_same_name_alias"], false,
        "unloaded self in another model space cannot establish an alias"
    );
}

#[tokio::test]
async fn naming_top_level_hash_analysis_retains_playable_source_identity() {
    let p = pool().await;
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("audio.wav");
    std::fs::write(&path, b"fictional source identity fixture").unwrap();
    let hash = runs::audio_hash(path).await.unwrap();
    let mut a = analysis();
    a.audio_sha256 = Some(hash.clone());
    a.embeddings.push(reference_voice(vec![1., 0.]));
    let m = meeting(&p, &a).await;
    sqlx::query("UPDATE meetings SET folder_path=? WHERE id=?")
        .bind(folder.path().to_string_lossy().as_ref())
        .bind(&m)
        .execute(&p)
        .await
        .unwrap();
    let (profile, _) = enrollment::confirm_profile(
        &p,
        &m,
        "Speaker 1",
        "Fictional top-level source",
        None,
        None,
    )
    .await
    .unwrap()
    .unwrap();
    let entry = enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .remove(0);
    let sample = enrollment::list_samples(&p, &profile, &entry.space_id)
        .await
        .unwrap()
        .remove(0);
    assert_eq!(sample.audio_sha256.as_deref(), Some(hash.as_str()));
    assert_eq!(
        enrollment::source_playback(&p, &sample.id)
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn live_automatic_name_is_displayed_but_never_protected_or_enrolled() {
    use crate::database::repositories::{
        transcript::TranscriptsRepository, voice_profile::VoiceProfilesRepository,
    };
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    let profile = VoiceProfilesRepository::create(&p, "Fictional automatic guess", None, &[], 0)
        .await
        .unwrap();
    let mut automatic = a.transcript_segments(None).await.unwrap().remove(0);
    automatic.speaker = Some("Fictional automatic guess".into());
    automatic.voice_profile_id = Some(profile.clone());
    TranscriptsRepository::upsert_transcript_segments(&p, &[(m.clone(), automatic)])
        .await
        .unwrap();
    let row: (String, String, String) = sqlx::query_as(
        "SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(
        row,
        (
            "Fictional automatic guess".into(),
            profile.clone(),
            "machine".into()
        )
    );
    let refined = TranscriptsRepository::update_speakers_by_sequence(
        &p,
        &m,
        &[(
            0,
            "Fictional automatic guess".into(),
            Some(profile.clone()),
            "Fictional automatic guess".into(),
            Some(profile.clone()),
        )],
    )
    .await
    .unwrap();
    assert_eq!(
        refined, 1,
        "automatic refinement remains available for an actual machine attribution"
    );
    let off = service::identify_speakers(
        &p,
        &m,
        service::IdentificationOptions {
            identification: config::Identification::Off,
            target_id: "not-needed".into(),
            replace_confirmed: false,
        },
        &std::sync::atomic::AtomicBool::new(false),
        |_, _, _| panic!("Off requires no inference"),
    )
    .await
    .unwrap();
    assert!(!off.suggestions[0].confirmed);
    assert!(off.suggestions[0].profile_id.is_none());
    // Even a stale group decision cannot authorize learning from a known live
    // guess. Only the explicit confirmation path may upgrade its row origin.
    sqlx::query(
        "UPDATE speaker_identity_decisions SET profile_id=?,confirmed=1 WHERE meeting_id=?",
    )
    .bind(&profile)
    .bind(&m)
    .execute(&p)
    .await
    .unwrap();
    let error = enrollment::enroll_from_meeting(
        &p,
        &profile,
        &m,
        "Fictional automatic guess",
        config::Identification::Campplus,
        "missing-target",
        &std::sync::atomic::AtomicBool::new(false),
        |_, _, _| panic!("an automatic label must not start extraction"),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("Confirm this speaker"));
    assert!(enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .is_empty());
    let mut redetected = a.clone();
    redetected.settings["request_id"] = "speaker-only-after-live-guess".into();
    redetected.settings["operation"] = "speaker_analysis".into();
    redetected.turns[0].speaker = "Speaker 2".into();
    service::publish_speaker_analysis(
        &p,
        &m,
        redetected,
        &std::sync::atomic::AtomicBool::new(false),
    )
    .await
    .unwrap();
    let regrouped: (String, Option<String>, String) = sqlx::query_as(
        "SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(regrouped, ("Speaker 2".into(), None, "machine".into()));
    a.settings["request_id"] = "final-after-unconfirmed-live-name".into();
    a.segments[0].text = "Fictional final words.".into();
    TranscriptsRepository::replace_transcripts_with_analysis(
        &p,
        &m,
        &a.transcript_segments(None).await.unwrap(),
        Some(&a),
    )
    .await
    .unwrap();
    let row: (String, Option<String>, String) = sqlx::query_as(
        "SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(row, ("Speaker 1".into(), None, "machine".into()));
}

#[tokio::test]
async fn explicit_same_name_human_confirmation_survives_late_live_writes_refinement_and_final_processing(
) {
    use crate::database::repositories::{
        transcript::TranscriptsRepository, voice_profile::VoiceProfilesRepository,
    };
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    let profile =
        VoiceProfilesRepository::create(&p, "Fictional same-name confirmation", None, &[], 0)
            .await
            .unwrap();
    let other = VoiceProfilesRepository::create(&p, "Fictional later wrong guess", None, &[], 0)
        .await
        .unwrap();
    let mut automatic = a.transcript_segments(None).await.unwrap().remove(0);
    automatic.speaker = Some("Fictional same-name confirmation".into());
    automatic.voice_profile_id = Some(profile.clone());
    TranscriptsRepository::upsert_transcript_segments(&p, &[(m.clone(), automatic.clone())])
        .await
        .unwrap();
    // This older proposal remains unconfirmed. The explicit per-row human
    // origin must take precedence without silently confirming a whole group.
    sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?, ?,0,NULL,datetime('now'))")
        .bind(&m)
        .bind("example-job")
        .bind("Speaker 1")
        .bind(&profile)
        .execute(&p)
        .await
        .unwrap();
    let row_id: String = sqlx::query_scalar("SELECT id FROM transcripts WHERE meeting_id=?")
        .bind(&m)
        .fetch_one(&p)
        .await
        .unwrap();
    TranscriptsRepository::set_segment_speaker(
        &p,
        &m,
        &row_id,
        "Fictional same-name confirmation",
        Some(&profile),
    )
    .await
    .unwrap();
    assert!(service::current_identity_suggestions(&p, &m).await.unwrap()[0].confirmed);
    automatic.speaker = Some("Fictional later wrong guess".into());
    automatic.voice_profile_id = Some(other.clone());
    TranscriptsRepository::upsert_transcript_segments(&p, &[(m.clone(), automatic)])
        .await
        .unwrap();
    let changed = TranscriptsRepository::update_speakers_by_sequence(
        &p,
        &m,
        &[(
            0,
            "Fictional same-name confirmation".into(),
            Some(profile.clone()),
            "Fictional later wrong guess".into(),
            Some(other),
        )],
    )
    .await
    .unwrap();
    assert_eq!(
        changed, 0,
        "same-value human confirmation blocks an otherwise matching late refinement"
    );
    let before: (String, String, String) = sqlx::query_as(
        "SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(
        before,
        (
            "Fictional same-name confirmation".into(),
            profile.clone(),
            "human".into()
        )
    );
    a.settings["request_id"] = "final-after-human-confirmation".into();
    a.segments[0].text = "Fictional final words.".into();
    TranscriptsRepository::replace_transcripts_with_analysis(
        &p,
        &m,
        &a.transcript_segments(None).await.unwrap(),
        Some(&a),
    )
    .await
    .unwrap();
    let after: (String, String, String) = sqlx::query_as(
        "SELECT speaker,voice_profile_id,identity_origin FROM transcripts WHERE meeting_id=?",
    )
    .bind(&m)
    .fetch_one(&p)
    .await
    .unwrap();
    assert_eq!(before, after);
    assert!(service::current_identity_suggestions(&p, &m).await.unwrap()[0].confirmed);
}

#[tokio::test]
async fn explicit_cached_name_and_enrollment_confirmation_upgrade_machine_origin() {
    use crate::database::repositories::{
        transcript::TranscriptsRepository, voice_profile::VoiceProfilesRepository,
    };
    let p = pool().await;
    let mut a = analysis();
    a.embeddings.push(reference_voice(vec![1., 0.]));
    let m = meeting(&p, &a).await;
    let profile =
        VoiceProfilesRepository::create(&p, "Fictional explicitly accepted source", None, &[], 0)
            .await
            .unwrap();
    let mut automatic = a.transcript_segments(None).await.unwrap().remove(0);
    automatic.speaker = Some("Fictional explicitly accepted source".into());
    automatic.voice_profile_id = Some(profile.clone());
    TranscriptsRepository::upsert_transcript_segments(&p, &[(m.clone(), automatic)])
        .await
        .unwrap();
    assert!(enrollment::list_enrollments(&p, &profile)
        .await
        .unwrap()
        .is_empty());
    enrollment::confirm_profile(
        &p,
        &m,
        "Fictional explicitly accepted source",
        "Fictional explicitly accepted source",
        None,
        Some(&profile),
    )
    .await
    .unwrap()
    .unwrap();
    let origin: String =
        sqlx::query_scalar("SELECT identity_origin FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(origin, "human");
    assert_eq!(
        enrollment::list_enrollments(&p, &profile).await.unwrap()[0].sample_count,
        1
    );
    // Simulate a separate known machine attribution. Explicit source acceptance
    // also upgrades it even though the displayed name/profile remain identical.
    sqlx::query("UPDATE transcripts SET identity_origin='machine' WHERE meeting_id=?")
        .bind(&m)
        .execute(&p)
        .await
        .unwrap();
    enrollment::confirm_enrollment_source(
        &p,
        &profile,
        &m,
        "Fictional explicitly accepted source",
        Some("example-job"),
    )
    .await
    .unwrap();
    assert_eq!(
        sqlx::query_scalar::<_, String>(
            "SELECT identity_origin FROM transcripts WHERE meeting_id=?"
        )
        .bind(&m)
        .fetch_one(&p)
        .await
        .unwrap(),
        "human"
    );
}

#[tokio::test]
async fn legacy_origin_default_stays_unknown_and_protected_during_reprocessing() {
    use crate::database::repositories::transcript::TranscriptsRepository;
    let p = pool().await;
    let mut a = analysis();
    let m = meeting(&p, &a).await;
    sqlx::query("DELETE FROM transcripts WHERE meeting_id=?")
        .bind(&m)
        .execute(&p)
        .await
        .unwrap();
    // Historical inserts do not know the new column. Its migration default
    // must preserve that uncertainty instead of claiming machine or human.
    sqlx::query("INSERT INTO transcripts(id,meeting_id,transcript,timestamp,speaker,audio_start_time,audio_end_time) VALUES('fictional-legacy-row',?,'Fictional old words.',datetime('now'),'Fictional historical name',1,5)")
        .bind(&m).execute(&p).await.unwrap();
    assert_eq!(
        sqlx::query_scalar::<_, String>(
            "SELECT identity_origin FROM transcripts WHERE meeting_id=?"
        )
        .bind(&m)
        .fetch_one(&p)
        .await
        .unwrap(),
        "legacy_unknown"
    );
    a.settings["request_id"] = "reprocess-with-legacy-authority".into();
    TranscriptsRepository::replace_transcripts_with_analysis(
        &p,
        &m,
        &a.transcript_segments(None).await.unwrap(),
        Some(&a),
    )
    .await
    .unwrap();
    let row: (String, String) =
        sqlx::query_as("SELECT speaker,identity_origin FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(
        row,
        ("Fictional historical name".into(), "legacy_unknown".into())
    );
    let mut late = a.transcript_segments(None).await.unwrap().remove(0);
    late.speaker = Some("Fictional automatic replacement".into());
    TranscriptsRepository::upsert_transcript_segments(&p, &[(m.clone(), late)])
        .await
        .unwrap();
    assert_eq!(
        TranscriptsRepository::update_speakers_by_sequence(
            &p,
            &m,
            &[(
                0,
                "Fictional historical name".into(),
                None,
                "Fictional automatic replacement".into(),
                None
            )]
        )
        .await
        .unwrap(),
        0
    );
    let stable: (String, String) =
        sqlx::query_as("SELECT speaker,identity_origin FROM transcripts WHERE meeting_id=?")
            .bind(&m)
            .fetch_one(&p)
            .await
            .unwrap();
    assert_eq!(stable, row);
}

#[tokio::test]
async fn integrity_progress_preserves_hash_and_reports_measured_completion() {
    use sha2::Digest;
    let folder = tempfile::tempdir().unwrap();
    let path = folder.path().join("fictional-model.bin");
    // Cross several read boundaries so progress must reflect intermediate bytes.
    let bytes: Vec<u8> = (0..(3 * 64 * 1024 + 17)).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, &bytes).unwrap();
    let expected = format!("{:x}", sha2::Sha256::digest(&bytes));
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = events.clone();
    let digest = runs::audio_hash_with_progress(path.clone(), move |percent| {
        recorded.lock().unwrap().push(percent);
    })
    .await
    .unwrap();
    assert_eq!(digest, expected);
    assert_eq!(runs::audio_hash(path).await.unwrap(), expected);
    let events = events.lock().unwrap();
    assert_eq!(events.first(), Some(&0));
    assert_eq!(events.last(), Some(&100));
    assert!(events.windows(2).all(|pair| pair[0] < pair[1]));
    assert_eq!(&*events, &[0, 33, 66, 99, 100]);
}

#[tokio::test]
async fn integrity_progress_finishes_empty_files_but_never_claims_failed_reads_complete() {
    let folder = tempfile::tempdir().unwrap();
    let empty = folder.path().join("empty.bin");
    std::fs::write(&empty, []).unwrap();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = events.clone();
    let digest = runs::audio_hash_with_progress(empty, move |percent| {
        recorded.lock().unwrap().push(percent);
    })
    .await
    .unwrap();
    assert_eq!(
        digest,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(&*events.lock().unwrap(), &[0, 100]);

    events.lock().unwrap().clear();
    let recorded = events.clone();
    assert!(
        runs::audio_hash_with_progress(folder.path().join("missing.bin"), move |percent| {
            recorded.lock().unwrap().push(percent);
        })
        .await
        .is_err()
    );
    assert!(events.lock().unwrap().is_empty());
}

#[tokio::test]
async fn readiness_progress_preserves_missing_target_error_before_model_checks() {
    let p = pool().await;
    let config = config::PipelineConfig {
        target_id: "missing-fictional-target".into(),
        ..Default::default()
    };
    let legacy_error = service::check_readiness(&p, &config).await.unwrap_err();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let recorded = events.clone();
    let error = service::check_readiness_with_progress(&p, &config, move |message, percent| {
        recorded.lock().unwrap().push((message.to_owned(), percent));
    })
    .await
    .unwrap_err();
    assert_eq!(error.to_string(), legacy_error.to_string());
    assert_eq!(
        *events.lock().unwrap(),
        vec![("Checking target connection…".to_owned(), None)]
    );
}

#[test]
fn native_word_alignment_splits_interruptions_and_preserves_untimed_chunks() {
    let mut a = analysis();
    a.duration = 10.;
    a.turns = serde_json::from_value(serde_json::json!([
        {"start":0.,"end":10.,"speaker":"Speaker 1"},
        {"start":4.,"end":5.,"speaker":"Speaker 2"}
    ])).unwrap();
    a.segments = serde_json::from_value(serde_json::json!([
        {"start":0.,"end":7.,"text":" Hello yes again"},
        {"start":7.,"end":10.,"text":" Keep this too"}
    ])).unwrap();
    a.words = serde_json::from_value(serde_json::json!([
        {"start":3.,"end":3.5,"text":" Hello"},
        {"start":4.1,"end":4.8,"text":" yes"},
        {"start":6.,"end":6.5,"text":" again"}
    ])).unwrap();
    native::annotate(&mut a);
    assert_eq!(a.segments.len(), 4);
    assert_eq!(a.segments[1].speaker.as_deref(), Some("Speaker 2"));
    assert_eq!(a.segments[1].overlapping_speakers.len(), 2);
    assert_eq!(a.segments.iter().map(|s| s.text.as_str()).collect::<String>(), " Hello yes again Keep this too");
    assert_eq!((a.segments[3].start, a.segments[3].end), (7., 10.));
}

fn acoustic_evidence_fixture() -> types::Analysis {
    let mut a = analysis();
    let mut second = a.segments[0].clone();
    second.start = 6.;
    second.end = 8.;
    a.segments.push(second);
    let mut e = reference_voice(vec![1., 0.]);
    e.source_intervals[0].start = 0.5;
    e.source_intervals[0].end = 9.5;
    e.clean_seconds = 9.;
    a.embeddings.push(e);
    a
}

#[tokio::test]
async fn naming_a_whole_group_keeps_acoustic_evidence_beyond_word_boundaries() {
    let p = pool().await;
    let a = acoustic_evidence_fixture();
    let m = meeting(&p, &a).await;
    let (profile, _) = enrollment::confirm_profile(&p, &m, "Speaker 1", "Fictional speaker", None, None)
        .await.unwrap().unwrap();
    let evidence = enrollment::list_enrollments(&p, &profile).await.unwrap();
    assert_eq!(evidence.len(), 1);
    let samples = enrollment::list_samples(&p, &profile, &evidence[0].space_id).await.unwrap();
    assert_eq!(samples[0].intervals[0].start, 0.5);
    assert_eq!(samples[0].intervals[0].end, 9.5);
}

#[tokio::test]
async fn partial_naming_and_overlap_never_authorize_the_whole_cached_voice() {
    for overlap in [false, true] {
        let p = pool().await;
        let mut a = acoustic_evidence_fixture();
        if overlap {
            a.turns.push(types::TimedSpeaker { start: 8.5, end: 9., speaker: "Speaker 2".into() });
            a.overlaps.push(types::Overlap { start: 8.5, end: 9., speakers: vec!["Speaker 1".into(), "Speaker 2".into()] });
        }
        let m = meeting(&p, &a).await;
        if !overlap {
            sqlx::query("UPDATE transcripts SET speaker='Corrected other voice' WHERE meeting_id=? AND audio_start_time=6")
                .bind(&m).execute(&p).await.unwrap();
        }
        let (profile, _) = enrollment::confirm_profile(&p, &m, "Speaker 1", "Fictional partial speaker", None, None)
            .await.unwrap().unwrap();
        assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
        enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
        assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
    }
}

#[tokio::test]
async fn recover_only_confirmed_current_evidence_once_and_respect_deletion() {
    let p = pool().await;
    let a = acoustic_evidence_fixture();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p, "Fictional previously named speaker", None, &[], 0).await.unwrap();
    sqlx::query("UPDATE transcripts SET voice_profile_id=?,identity_origin='human' WHERE meeting_id=?")
        .bind(&profile).bind(&m).execute(&p).await.unwrap();
    // A name alone is insufficient without an explicit confirmation in this run.
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
    sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?, ?,1,NULL,datetime('now'))")
        .bind(&m).bind("old-run").bind("Speaker 1").bind(&profile).execute(&p).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
    sqlx::query("UPDATE speaker_identity_decisions SET group_version='example-job' WHERE meeting_id=?")
        .bind(&m).execute(&p).await.unwrap();
    sqlx::query("UPDATE transcripts SET identity_origin='machine' WHERE meeting_id=? AND audio_start_time=6")
        .bind(&m).execute(&p).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
    sqlx::query("UPDATE transcripts SET identity_origin='human' WHERE meeting_id=?")
        .bind(&m).execute(&p).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    let evidence = enrollment::list_enrollments(&p, &profile).await.unwrap();
    assert_eq!(evidence.len(), 1);
    assert_eq!(evidence[0].sample_count, 1);
    enrollment::delete_enrollment(&p, &profile, &evidence[0].space_id).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    assert!(enrollment::list_enrollments(&p, &profile).await.unwrap().is_empty());
}

#[test]
fn community_identity_uses_lower_baseline_but_preserves_ambiguity_and_context() {
    use crate::speaker_diarization::matching_policy as policy;
    let threshold = policy::saved_threshold("wespeaker", false, None);
    assert_eq!(threshold, 0.60);
    assert_eq!(policy::saved_choice(&[(0.70, threshold), (0.35, threshold)]), Some(0));
    assert_eq!(policy::saved_choice(&[(0.59, threshold), (0.35, threshold)]), None);
    assert_eq!(policy::saved_choice(&[(0.70, threshold), (0.65, threshold)]), None);
    assert_eq!(policy::saved_threshold("wespeaker", false, Some(true)), 0.60);
    assert_eq!(policy::saved_threshold("wespeaker", false, Some(false)), 0.75);
    assert_eq!(policy::saved_threshold("campplus", false, None), 0.75);
    assert_eq!(policy::saved_baseline("unknown"), 0.80);
}

#[tokio::test]
async fn confirmation_learns_each_meeting_once_and_recovery_preserves_removed_sources() {
    let p = pool().await;
    let mut a = acoustic_evidence_fixture();
    let first = meeting(&p, &a).await;
    let (profile, _) = enrollment::confirm_profile(&p, &first, "Speaker 1", "Fictional voice", None, None)
        .await.unwrap().unwrap();
    a.settings["request_id"] = "second-job".into();
    let second = meeting(&p, &a).await;
    let cancel = std::sync::atomic::AtomicBool::new(false);
    service::identify_speakers(&p, &second, identification_options(false), &cancel, |_, _, _| {})
        .await.unwrap();
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples").fetch_one(&p).await.unwrap();
    assert_eq!(count, 1, "automatic suggestions cannot train profiles");
    service::confirm_identity_suggestion(&p, &second, "second-job", "Speaker 1", &profile).await.unwrap();
    service::confirm_identity_suggestion(&p, &second, "second-job", "Speaker 1", &profile).await.unwrap();
    let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM enrollment_samples").fetch_one(&p).await.unwrap();
    assert_eq!(count, 2, "new meetings contribute, repeated confirmations do not");
    let removed: String = sqlx::query_scalar("SELECT id FROM enrollment_samples WHERE source_meeting_id=?")
        .bind(&first).fetch_one(&p).await.unwrap();
    enrollment::remove_sample(&p, &removed).await.unwrap();
    // Simulate an old confirmation that updated labels but omitted enrollment.
    a.settings["request_id"] = "third-job".into();
    let third = meeting(&p, &a).await;
    sqlx::query("UPDATE transcripts SET voice_profile_id=?,identity_origin='human' WHERE meeting_id=?")
        .bind(&profile).bind(&third).execute(&p).await.unwrap();
    sqlx::query("INSERT INTO speaker_identity_decisions VALUES(?,?,?, ?,1,NULL,datetime('now'))")
        .bind(&third).bind("third-job").bind("Speaker 1").bind(&profile).execute(&p).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &profile).await.unwrap();
    let sources: Vec<String> = sqlx::query_scalar("SELECT source_meeting_id FROM enrollment_samples WHERE profile_id=?")
        .bind(&profile).fetch_all(&p).await.unwrap();
    assert_eq!(sources.len(), 2);
    assert!(sources.contains(&second) && sources.contains(&third));
    assert!(!sources.contains(&first), "deleted evidence must stay deleted");
}

#[tokio::test]
async fn confirmation_enrolls_alternate_cached_embedding_space() {
    let p = pool().await;
    let a = acoustic_evidence_fixture();
    let m = meeting(&p, &a).await;
    let profile = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p, "Fictional voice", None, &[], 0).await.unwrap();
    let mut alternate = a.embeddings[0].clone();
    alternate.embedding_space.as_mut().unwrap().id = "fictional-alternate-space".into();
    sqlx::query("INSERT INTO speaker_group_embeddings VALUES(?,?,?,?,?,?)")
        .bind(&m).bind("example-job").bind("Speaker 1").bind("fictional-alternate-space")
        .bind(serde_json::to_string(&alternate).unwrap()).bind("{}")
        .execute(&p).await.unwrap();
    service::confirm_identity_suggestion(&p, &m, "example-job", "Speaker 1", &profile).await.unwrap();
    assert_eq!(enrollment::list_enrollments(&p, &profile).await.unwrap().len(), 2);
}

#[tokio::test]
async fn replacing_identity_moves_only_that_meetings_evidence() {
    let p = pool().await;
    let mut a = acoustic_evidence_fixture();
    let first = meeting(&p, &a).await;
    let (old, _) = enrollment::confirm_profile(&p, &first, "Speaker 1", "Fictional old identity", None, None)
        .await.unwrap().unwrap();
    a.settings["request_id"] = "second-job".into();
    let second = meeting(&p, &a).await;
    service::confirm_identity_suggestion(&p, &second, "second-job", "Speaker 1", &old).await.unwrap();
    let new = crate::database::repositories::voice_profile::VoiceProfilesRepository::create(
        &p, "Fictional new identity", None, &[], 0).await.unwrap();
    service::confirm_identity_replacement(&p, &second, "second-job", "Speaker 1", &new,
        service::IdentityAssignment { profile_id: Some(old.clone()), name: Some("Fictional old identity".into()) })
        .await.unwrap();
    enrollment::recover_confirmed_evidence(&p, &old).await.unwrap();
    let sources: Vec<(String, String)> = sqlx::query_as("SELECT profile_id,source_meeting_id FROM enrollment_samples")
        .fetch_all(&p).await.unwrap();
    assert_eq!(sources.len(), 2);
    assert!(sources.contains(&(old, first)));
    assert!(sources.contains(&(new, second)));
}
