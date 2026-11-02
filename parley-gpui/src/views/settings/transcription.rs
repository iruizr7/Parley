//! Final processing defaults; the shared editor also backs meeting overrides.
use super::SettingsView;
use gpui_kit::component::{
    Icon,
    setting::{SettingGroup, SettingItem, SettingPage},
};
use gpui_kit::*;

pub fn page(
    pipeline: &Entity<crate::views::processing::PipelineEditor>,
    _cx: &mut Context<SettingsView>,
) -> SettingPage {
    let pipeline = pipeline.clone();
    SettingPage::new("Transcription")
        .icon(Icon::new(gpui_kit::assets::IconName::Mic))
        .group(SettingGroup::new().title("Final processing pipeline").item(
            SettingItem::render(move |_, _, _| pipeline.clone().into_any_element()).keywords([
                "transcription",
                "target",
                "speech engine",
                "speech model",
                "speaker pipeline",
                "speaker identification",
                "CAMPPlus",
                "WeSpeaker",
                "whisper.cpp",
                "faster-whisper",
                "MLX Whisper",
            ]),
        ))
}
