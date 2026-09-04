from __future__ import annotations

import pytest

from video_digest.sources.youtube import VideoRejected, parse_info

VID = "dQw4w9WgXcQ"


def _base_info(**overrides: object) -> dict[str, object]:
    info: dict[str, object] = {
        "id": VID,
        "title": "A Video",
        "channel": "Some Channel",
        "channel_id": "UCxxxx",
        "duration": 300,
        "upload_date": "20260820",
        "description": "A description.",
        "chapters": [],
        "language": "en",
        "subtitles": {},
        "automatic_captions": {"en": [{"ext": "vtt"}]},
        "thumbnail": "https://i.ytimg.com/x.jpg",
        "live_status": "not_live",
        "availability": "public",
        "formats": [{"format_id": "18"}],
    }
    info.update(overrides)
    return info


class TestCleanExtraction:
    def test_maps_the_common_fields(self) -> None:
        meta = parse_info(_base_info(), max_duration_minutes=240)
        assert meta.video_id == VID
        assert meta.title == "A Video"
        assert meta.channel == "Some Channel"
        assert meta.duration_s == 300
        assert meta.upload_date == "2026-08-20"
        assert meta.auto_caption_langs == ["en"]
        assert meta.has_manual_subs is False

    def test_manual_subs_detected(self) -> None:
        meta = parse_info(
            _base_info(subtitles={"en": [{"ext": "vtt"}], "sv": [{"ext": "vtt"}]}),
            max_duration_minutes=240,
        )
        assert meta.has_manual_subs is True
        assert meta.manual_sub_langs == ["en", "sv"]

    def test_missing_upload_date_is_none(self) -> None:
        meta = parse_info(_base_info(upload_date=None), max_duration_minutes=240)
        assert meta.upload_date is None

    def test_uploader_falls_back_when_channel_absent(self) -> None:
        info = _base_info(uploader="Fallback Name")
        del info["channel"]
        meta = parse_info(info, max_duration_minutes=240)
        assert meta.channel == "Fallback Name"


class TestRejections:
    @pytest.mark.parametrize("status", ["is_live", "is_upcoming"])
    def test_live_in_progress(self, status: str) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(live_status=status), max_duration_minutes=240)
        assert exc.value.reason == "live_in_progress"

    def test_was_live_is_accepted(self) -> None:
        """A finished livestream is a normal, transcribable video."""
        meta = parse_info(_base_info(live_status="was_live"), max_duration_minutes=240)
        assert meta.video_id == VID

    @pytest.mark.parametrize("availability", ["premium_only", "subscriber_only"])
    def test_members_only(self, availability: str) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(availability=availability), max_duration_minutes=240)
        assert exc.value.reason == "members_only"

    def test_age_gated_without_credentials(self) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(availability="needs_auth"), max_duration_minutes=240)
        assert exc.value.reason == "age_gated"

    def test_private(self) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(availability="private"), max_duration_minutes=240)
        assert exc.value.reason == "private"

    def test_region_blocked_no_playable_format(self) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(formats=[], url=None), max_duration_minutes=240)
        assert exc.value.reason == "region_blocked"

    def test_over_max_duration(self) -> None:
        with pytest.raises(VideoRejected) as exc:
            parse_info(_base_info(duration=4 * 3600 + 1), max_duration_minutes=240)
        assert exc.value.reason == "too_long"

    def test_exactly_at_max_duration_is_accepted(self) -> None:
        meta = parse_info(_base_info(duration=4 * 3600), max_duration_minutes=240)
        assert meta.duration_s == 4 * 3600
