from __future__ import annotations

import pytest

from video_digest.config import AcquisitionConfig
from video_digest.sources.youtube import VideoMetadata
from video_digest.transcripts.ladder import NeedsASR, acquire, t1_is_trustworthy

VID = "dQw4w9WgXcQ"

#: A plausible-density VTT: ~30 words over 60s of declared duration is 30
#: tokens/minute — comfortably above MIN_TOKENS_PER_MINUTE (40) once repeated
#: across enough cues. Kept simple: one long cue.
_GOOD_VTT = (
    "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n"
    + "word " * 40
    + "\n"
)
_SPARSE_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nonly a few words\n"
#: 25 words over the fixture's 60s duration = 25 tokens/min: below the
#: strict floor (40) but above the loosened one (20).
_BORDERLINE_VTT = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n" + "word " * 25 + "\n"


def _meta(**overrides: object) -> VideoMetadata:
    fields: dict[str, object] = {
        "video_id": VID,
        "title": "A Video",
        "channel": "A Channel",
        "channel_id": "UCxxxx",
        "duration_s": 60,
        "upload_date": "2026-08-20",
        "description": "d",
        "language": "en",
        "manual_sub_langs": [],
        "auto_caption_langs": [],
    }
    fields.update(overrides)
    return VideoMetadata(**fields)  # type: ignore[arg-type]


@pytest.fixture
def cfg() -> AcquisitionConfig:
    return AcquisitionConfig(subtitle_languages=["en"])


class TestT0PreferredOverT1:
    def test_manual_subs_used_when_present(self, cfg: AcquisitionConfig) -> None:
        calls: list[tuple[str, str, bool]] = []

        def fetch(video_id: str, lang: str, auto: bool) -> str:
            calls.append((video_id, lang, auto))
            return _GOOD_VTT

        meta = _meta(manual_sub_langs=["en"], auto_caption_langs=["en"])
        result = acquire(meta, cfg, _fetch=fetch)

        assert result.tier == "T0"
        assert calls == [(VID, "en", False)]  # never asked for auto-captions


class TestT1FallsBackWhenNoManualSubs:
    def test_auto_captions_used_and_tagged_t1(self, cfg: AcquisitionConfig) -> None:
        meta = _meta(auto_caption_langs=["en"])
        result = acquire(meta, cfg, _fetch=lambda vid, lang, auto: _GOOD_VTT)
        assert result.tier == "T1"
        assert result.degraded is False

    def test_loosen_accepts_a_borderline_track_the_strict_heuristic_would_reject(
        self, cfg: AcquisitionConfig
    ) -> None:
        """A thin-but-not-empty track only clears the floor when the ASR
        worker is confirmed offline (M5's degrade_to_captions_when_offline)."""
        meta = _meta(auto_caption_langs=["en"])
        with pytest.raises(NeedsASR):
            acquire(meta, cfg, _fetch=lambda *a: _BORDERLINE_VTT)

        result = acquire(meta, cfg, loosen=True, _fetch=lambda *a: _BORDERLINE_VTT)
        assert result.tier == "T1"
        assert result.degraded is True

    def test_language_preference_order_is_respected(self, cfg: AcquisitionConfig) -> None:
        cfg = AcquisitionConfig(subtitle_languages=["sv", "en"])
        calls: list[str] = []

        def fetch(video_id: str, lang: str, auto: bool) -> str:
            calls.append(lang)
            return _GOOD_VTT

        meta = _meta(auto_caption_langs=["en", "sv"], language="sv")
        acquire(meta, cfg, _fetch=fetch)
        assert calls == ["sv"]  # "sv" preferred over "en", and it is available


class TestNoUsableCaptions:
    def test_no_tracks_at_all_needs_asr(self, cfg: AcquisitionConfig) -> None:
        meta = _meta()
        with pytest.raises(NeedsASR) as exc:
            acquire(meta, cfg, _fetch=lambda *a: _GOOD_VTT)
        assert exc.value.reason == "no_captions_available"

    def test_low_quality_auto_captions_need_asr(self, cfg: AcquisitionConfig) -> None:
        meta = _meta(auto_caption_langs=["en"])
        with pytest.raises(NeedsASR) as exc:
            acquire(meta, cfg, _fetch=lambda *a: _SPARSE_VTT)
        assert exc.value.reason == "low_token_density"

    def test_machine_translated_track_needs_asr(self, cfg: AcquisitionConfig) -> None:
        # Video's declared language is "de"; the "en" auto-caption available
        # is therefore a translation of the German ASR, not a transcription.
        meta = _meta(auto_caption_langs=["en"], language="de")
        with pytest.raises(NeedsASR) as exc:
            acquire(meta, cfg, _fetch=lambda *a: _GOOD_VTT)
        assert exc.value.reason == "machine_translated"

    def test_a_region_code_on_the_declared_language_is_not_a_mismatch(
        self, cfg: AcquisitionConfig
    ) -> None:
        """Regression from the pilot run against a real video: yt-dlp
        reported the video's declared language as "en-US" while the genuine
        (non-translated) auto-caption track is coded plain "en" — an
        exact-string compare rejected it as machine_translated, which would
        misfire on most real YouTube videos carrying a region-coded
        language."""
        meta = _meta(auto_caption_langs=["en"], language="en-US")
        result = acquire(meta, cfg, _fetch=lambda *a: _GOOD_VTT)
        assert result.tier == "T1"


class TestT1Heuristic:
    def test_matching_language_and_good_density_is_trustworthy(self) -> None:
        trustworthy, reason = t1_is_trustworthy(
            requested_lang="en", declared_language="en", cue_text="word " * 100, duration_s=60
        )
        assert trustworthy is True
        assert reason is None

    def test_region_coded_declared_language_matches_the_bare_caption_code(self) -> None:
        trustworthy, reason = t1_is_trustworthy(
            requested_lang="en", declared_language="en-US", cue_text="word " * 100, duration_s=60
        )
        assert trustworthy is True
        assert reason is None

    def test_genuinely_different_language_is_still_rejected(self) -> None:
        trustworthy, reason = t1_is_trustworthy(
            requested_lang="en", declared_language="de-DE", cue_text="word " * 100, duration_s=60
        )
        assert trustworthy is False
        assert reason == "machine_translated"

    def test_no_declared_language_is_not_treated_as_a_mismatch(self) -> None:
        """Many videos have no declared language at all — absence must not
        be read as a translation signal."""
        trustworthy, _reason = t1_is_trustworthy(
            requested_lang="en", declared_language=None, cue_text="word " * 100, duration_s=60
        )
        assert trustworthy is True

    def test_loosen_halves_the_density_threshold(self) -> None:
        # 100 words over 300s = 20 tokens/min: below the normal 40 floor,
        # above the loosened 20 floor.
        text = "word " * 100
        strict, _ = t1_is_trustworthy(
            requested_lang="en", declared_language="en", cue_text=text, duration_s=300
        )
        loose, _ = t1_is_trustworthy(
            requested_lang="en",
            declared_language="en",
            cue_text=text,
            duration_s=300,
            loosen=True,
        )
        assert strict is False
        assert loose is True

    def test_loosen_does_not_override_a_language_mismatch(self) -> None:
        """Degrading the density floor when the ASR worker is offline (M5)
        is not the same as accepting a translated track — those stay
        rejected regardless."""
        trustworthy, reason = t1_is_trustworthy(
            requested_lang="en",
            declared_language="de",
            cue_text="word " * 100,
            duration_s=60,
            loosen=True,
        )
        assert trustworthy is False
        assert reason == "machine_translated"
