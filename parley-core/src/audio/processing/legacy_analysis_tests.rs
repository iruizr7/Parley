use super::config::NATIVE_EMBEDDING;
use super::{
    enrollment::confirm_profile, identification::match_profiles_for_meeting, native::annotate,
    runs::load, types::*,
};
#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::repositories::{
        transcript::TranscriptsRepository, voice_profile::VoiceProfilesRepository,
    };
    use sqlx::SqlitePool;

    fn analysis() -> Analysis {
        serde_json::from_value(serde_json::json!({"version":1,"duration":10.,"language":"es",
            "segments":[{"start":0.,"end":10.,"text":"Hola","speaker":"Speaker 1"}],
            "words":[],"turns":[{"start":0.,"end":10.,"speaker":"Speaker 1"}],"overlaps":[],
            "embeddings":[{"speaker":"Speaker 1","model_id":"wespeaker-checkpoint-a","vector":[1.,0.],"clean_seconds":10.}],
            "settings":{"recognizer":"faster","diarization":"community"}})).unwrap()
    }
    async fn pool() -> SqlitePool {
        let p = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&p).await.unwrap();
        p
    }
    async fn save(p: &SqlitePool, a: &Analysis) -> String {
        let segments = a.transcript_segments(None).await.unwrap();
        TranscriptsRepository::save_transcript_with_analysis(p, "Test", &segments, None, Some(a))
            .await
            .unwrap()
    }

    #[test]
    fn invalid_engine_output_cannot_be_saved() {
        let mut a = analysis();
        a.segments[0].end = 100.;
        assert!(a.validate().is_err());
        a = analysis();
        a.embeddings[0].vector[0] = f32::NAN;
        assert!(a.validate().is_err());
        a = analysis();
        a.embeddings[0].clean_seconds = 0.5;
        assert!(a.validate().is_err());
    }
    #[test]
    fn short_interjection_and_overlap_are_preserved() {
        let mut a = analysis();
        // Alignment only uses word timings when they cover the source text.
        a.segments[0].text = "Hola sí".into();
        a.turns.push(TimedSpeaker {
            start: 4.,
            end: 5.,
            speaker: "Speaker 2".into(),
        });
        a.words = vec![
            Word {
                start: 3.,
                end: 3.5,
                text: " Hola".into(),
                probability: None,
                speaker: None,
                overlapping_speakers: vec![],
            },
            Word {
                start: 4.1,
                end: 4.8,
                text: " sí".into(),
                probability: None,
                speaker: None,
                overlapping_speakers: vec![],
            },
        ];
        annotate(&mut a);
        assert_eq!(a.segments.len(), 2);
        assert_eq!(a.segments[1].speaker.as_deref(), Some("Speaker 2"));
        assert_eq!(a.overlapping_speakers(Some(4.2), Some(4.3)).len(), 2);
        assert!(a.overlapping_speakers(Some(5.), Some(6.)).is_empty());
    }
    #[tokio::test]
    async fn confirmed_voice_survives_restart_and_inference_never_trains_it() {
        let p = pool().await;
        let a = analysis();
        let meeting = save(&p, &a).await;
        assert!(match_profiles_for_meeting(&p, "", &a.embeddings)
            .await
            .unwrap()
            .is_empty());
        let (id, changed) = confirm_profile(&p, &meeting, "Speaker 1", "Ana", None, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(changed, 1);
        let profile = VoiceProfilesRepository::get_by_id(&p, &id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            profile.embedding_dim, 0,
            "WeSpeaker must not masquerade as CAM++"
        );
        let loaded = load(&p, &meeting).await.unwrap().unwrap();
        let rows = loaded.transcript_segments(Some(&p)).await.unwrap();
        assert!(
            rows[0].voice_profile_id.is_none(),
            "materializing analysis never identifies implicitly"
        );
        assert_eq!(rows[0].speaker.as_deref(), Some("Speaker 1"));
        let saved: (String, Option<String>) =
            sqlx::query_as("SELECT speaker,voice_profile_id FROM transcripts WHERE meeting_id=?")
                .bind(&meeting)
                .fetch_one(&p)
                .await
                .unwrap();
        assert_eq!(saved, ("Ana".into(), Some(id)));
        let seconds: f64 = sqlx::query_scalar("SELECT clean_seconds FROM voice_profile_embeddings")
            .fetch_one(&p)
            .await
            .unwrap();
        assert_eq!(seconds, 10.);
    }
    #[tokio::test]
    async fn identical_dimensions_do_not_mix_embedding_models() {
        let p = pool().await;
        let a = analysis();
        let meeting = save(&p, &a).await;
        confirm_profile(&p, &meeting, "Speaker 1", "Ana", None, None)
            .await
            .unwrap();
        let mut other = a.embeddings.clone();
        other[0].model_id = "wespeaker-checkpoint-b".into();
        assert!(match_profiles_for_meeting(&p, "", &other)
            .await
            .unwrap()
            .is_empty());
    }
    #[tokio::test]
    async fn confirmed_native_evidence_uses_window_count_not_seconds() {
        let p = pool().await;
        let mut a = analysis();
        a.embeddings.push(SpeakerEmbedding {
            speaker: "Speaker 1".into(),
            model_id: NATIVE_EMBEDDING.into(),
            vector: vec![0., 1., 0.],
            clean_seconds: 9.,
            sample_count: 3,
            embedding_space: None,
            source_intervals: vec![],
        });
        let meeting = save(&p, &a).await;
        let (id, _) = confirm_profile(&p, &meeting, "Speaker 1", "Ana", None, None)
            .await
            .unwrap()
            .unwrap();
        let profile = VoiceProfilesRepository::get_by_id(&p, &id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(profile.sample_count, 3);
        assert_eq!(profile.embedding_dim, 3);
        assert!(
            match_profiles_for_meeting(&p, "", &a.embeddings)
                .await
                .unwrap()
                .is_empty(),
            "unknown legacy spaces cannot identify"
        );
    }
    #[tokio::test]
    async fn ambiguous_voice_match_stays_unassigned() {
        let p = pool().await;
        let a = analysis();
        let meeting = save(&p, &a).await;
        confirm_profile(&p, &meeting, "Speaker 1", "Ana", None, None)
            .await
            .unwrap();
        let id = VoiceProfilesRepository::create(&p, "Similar voice", None, &[], 0)
            .await
            .unwrap();
        sqlx::query("INSERT INTO voice_profile_embeddings VALUES (?,?,?,10)")
            .bind(id)
            .bind("wespeaker-checkpoint-a")
            .bind("[0.99,0.1]")
            .execute(&p)
            .await
            .unwrap();
        assert!(match_profiles_for_meeting(&p, "", &a.embeddings)
            .await
            .unwrap()
            .is_empty());
    }
    #[tokio::test]
    async fn replacement_and_analysis_roll_back_together() {
        let p = pool().await;
        let a = analysis();
        let meeting = save(&p, &a).await;
        let mut invalid = a.clone();
        invalid.duration = -1.;
        let mut segments = a.transcript_segments(None).await.unwrap();
        segments[0].text = "replacement".into();
        assert!(TranscriptsRepository::replace_transcripts_with_analysis(
            &p,
            &meeting,
            &segments,
            Some(&invalid)
        )
        .await
        .is_err());
        let text: String =
            sqlx::query_scalar("SELECT transcript FROM transcripts WHERE meeting_id=?")
                .bind(&meeting)
                .fetch_one(&p)
                .await
                .unwrap();
        assert_eq!(text, "Hola");
        assert!(load(&p, &meeting).await.unwrap().is_some());
        TranscriptsRepository::replace_transcripts_for_meeting(&p, &meeting, &segments)
            .await
            .unwrap();
        assert!(
            load(&p, &meeting).await.unwrap().is_none(),
            "native reprocessing must clear stale overlap analysis"
        );
    }
    #[tokio::test]
    async fn merging_profiles_preserves_model_specific_evidence() {
        let p = pool().await;
        let a = analysis();
        let meeting = save(&p, &a).await;
        let (loser, _) = confirm_profile(&p, &meeting, "Speaker 1", "Ana", None, None)
            .await
            .unwrap()
            .unwrap();
        let winner = VoiceProfilesRepository::create(&p, "Ana existing", None, &[0., 1., 0.], 4)
            .await
            .unwrap();
        VoiceProfilesRepository::merge_into(&p, &winner, &loser)
            .await
            .unwrap();
        let matched = match_profiles_for_meeting(&p, "", &a.embeddings)
            .await
            .unwrap();
        assert!(
            matched.is_empty(),
            "merging must not invent a compatible embedding space"
        );
        let retained:String=sqlx::query_scalar("SELECT profile_id FROM voice_profile_embeddings WHERE model_id='wespeaker-checkpoint-a'").fetch_one(&p).await.unwrap();
        assert_eq!(retained, winner);
    }
}
