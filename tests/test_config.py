from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_digest.config import LLMConfig, LLMTierConfig, Settings, load_settings

#: The two required sections with no field defaults — every direct
#: `Settings(...)` construction in this file needs them, so it is a fixture
#: rather than repeated in every test.
_MIN_LLM = {
    "tiers": {
        "map": {"primary": {"provider": "ollama", "model": "qwen3.5:9b"}},
        "reduce": {"primary": {"provider": "ollama", "model": "qwen3.5:9b"}},
    }
}
_MIN_TRANSCRIPTION = {"remote_url": "http://mac.local:8000"}


class TestShippedConfig:
    def test_repo_config_yaml_is_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The shipped config.yaml must load — it is the deployment default."""
        monkeypatch.setenv("VIDEODIGEST_ADMIN_API_KEY", "test-key")
        # config.yaml's tiers both default to an openrouter primary now.
        monkeypatch.setenv("VIDEODIGEST_OPENROUTER_API_KEY", "sk-or-test")
        settings = load_settings(Path(__file__).parent.parent / "config.yaml")
        assert settings.vault.notes_dir == "13 video-summaries"
        assert settings.vault.transcripts_dir == "14 video-transcripts"
        assert settings.llm.tiers["map"].allow_cloud_fallback is True
        assert settings.llm.tiers["reduce"].allow_cloud_fallback is True
        assert settings.llm.tiers["map"].primary.provider.value == "openrouter"

    def test_unknown_key_is_a_startup_crash(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("vault:\n  notess_dir: oops\n")  # typo'd key
        with pytest.raises(ValidationError):
            load_settings(bad)


class TestVaultFolders:
    def test_empty_folder_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                llm=_MIN_LLM,
                transcription=_MIN_TRANSCRIPTION,
                vault={"notes_dir": "/"},
            )

    def test_empty_segment_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                llm=_MIN_LLM,
                transcription=_MIN_TRANSCRIPTION,
                vault={"notes_dir": "13 video-summaries//sub"},
            )

    def test_folder_slashes_stripped(self) -> None:
        settings = Settings(
            llm=_MIN_LLM,
            transcription=_MIN_TRANSCRIPTION,
            vault={"notes_dir": "/13 video-summaries/"},
        )
        assert settings.vault.notes_dir == "13 video-summaries"


class TestLLMTiers:
    def test_missing_required_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMConfig(tiers={"map": LLMTierConfig(**_MIN_LLM["tiers"]["map"])})

    def test_cloud_fallback_without_key_refuses_to_boot(self) -> None:
        llm = {
            "tiers": {
                "map": {"primary": {"provider": "ollama", "model": "x"}},
                "reduce": {
                    "primary": {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
                },
            }
        }
        with pytest.raises(ValidationError):
            Settings(llm=llm, transcription=_MIN_TRANSCRIPTION)

    def test_cloud_fallback_with_key_boots(self) -> None:
        llm = {
            "tiers": {
                "map": {"primary": {"provider": "ollama", "model": "x"}},
                "reduce": {
                    "primary": {"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"},
                },
            }
        }
        settings = Settings(
            llm=llm, transcription=_MIN_TRANSCRIPTION, openrouter_api_key="sk-or-test"
        )
        assert settings.llm.tiers["reduce"].primary.provider.value == "openrouter"

    def test_cloud_only_tier_with_fallback_disabled_refused(self) -> None:
        llm = {
            "tiers": {
                "map": {"primary": {"provider": "ollama", "model": "x"}},
                "reduce": {
                    "primary": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-haiku-4.5",
                    },
                    "allow_cloud_fallback": False,
                },
            }
        }
        with pytest.raises(ValidationError):
            Settings(llm=llm, transcription=_MIN_TRANSCRIPTION)


class TestTranscription:
    def test_remote_url_required(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm=_MIN_LLM, transcription={})

    def test_bad_url_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(llm=_MIN_LLM, transcription={"remote_url": "not-a-url"})


class TestNotifications:
    def test_enabled_without_target_refuses_to_boot(self) -> None:
        with pytest.raises(ValidationError):
            Settings(
                llm=_MIN_LLM,
                transcription=_MIN_TRANSCRIPTION,
                notifications={"enabled": True},
            )

    def test_enabled_with_target_boots(self) -> None:
        settings = Settings(
            llm=_MIN_LLM,
            transcription=_MIN_TRANSCRIPTION,
            notifications={"enabled": True, "ntfy_url": "http://ntfy.lan:8080", "topic": "vd"},
        )
        assert settings.notifications.enabled is True
