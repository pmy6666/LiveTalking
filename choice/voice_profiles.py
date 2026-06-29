from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VOICE_PROFILE_BY_AVATAR = {
    "avatar1": "DongQing_6s",
    "wav2lip256_avatar1": "DongQing_6s",
    "avatar2": "DongQing_6s",
    "avatar3": "DongQing_6s",
    "avatar4": "SaBeining",
    "avatar5": "SaBeining",
    "avatar6": "SaBeining",
    "avatar7": "DongQing_6s",
}


VOICE_FILES = {
    "DongQing_6s": PROJECT_ROOT / "bilibili_downloads" / "DongQing_6s.wav",
    "Female": PROJECT_ROOT / "bilibili_downloads" / "Female.wav",
    "SaBeining": PROJECT_ROOT / "bilibili_downloads" / "SaBeining.wav",
    "SaBeining_enhanced": PROJECT_ROOT / "bilibili_downloads" / "SaBeining_enhanced.wav",
}


VOICE_TEXT_ALIASES = {
    "SaBeining_enhanced": "SaBeining",
}


def load_voice_texts(content_path: Path = None) -> dict[str, str]:
    content_path = content_path or PROJECT_ROOT / "docs" / "notes" / "content.txt"
    if not content_path.exists():
        legacy_path = PROJECT_ROOT / "content.txt"
        if legacy_path.exists():
            content_path = legacy_path
    texts = {}
    if not content_path.exists():
        return texts
    for raw_line in content_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            name, text = line.split(":", 1)
        elif "：" in line:
            name, text = line.split("：", 1)
        else:
            continue
        text = text.strip().strip('"').strip("“”")
        texts[name.strip()] = text
    return texts


def resolve_voice_profile(avatar_id: str) -> dict[str, str]:
    profile_name = VOICE_PROFILE_BY_AVATAR.get(avatar_id)
    if not profile_name:
        return {}

    voice_file = VOICE_FILES.get(profile_name)
    voice_texts = load_voice_texts()
    voice_text = voice_texts.get(profile_name, "")
    if not voice_text:
        voice_text = voice_texts.get(VOICE_TEXT_ALIASES.get(profile_name, ""), "")
    if not voice_file or not voice_file.exists():
        return {}

    return {
        "profile": profile_name,
        "ref_file": str(voice_file),
        "ref_text": voice_text,
    }


def apply_voice_profile(opt, avatar_id: str):
    profile = resolve_voice_profile(avatar_id)
    if not profile:
        return profile
    opt.REF_FILE = profile["ref_file"]
    opt.REF_TEXT = profile["ref_text"]
    return profile
