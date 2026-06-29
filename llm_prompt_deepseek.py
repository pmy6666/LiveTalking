import json
import os
import time

from utils.logger import logger


DEFAULT_PROMPT = (
    "A front-facing digital human is speaking naturally, upper body visible, "
    "stable camera, realistic lighting."
)
DEFAULT_NEGATIVE_PROMPT = (
    "blur, low quality, distorted face, bad hands, extra fingers, deformed body, "
    "strange movement, jitter, flicker, motion blur, out of frame"
)

SYSTEM_PROMPT = (
    "You generate concise English prompts for EchoMimicV3 talking-head video generation. "
    "Return only valid JSON. The prompt must be short, direct, visual, and suitable "
    "for a realistic talking portrait. Do not describe complex actions, large body "
    "movement, camera movement, or scene changes. The negative_prompt must list "
    "visual defects to avoid."
)

USER_PROMPT_TEMPLATE = """Avatar:
{avatar_description}

Speech text:
{speech_text}

Scene:
{scene}

Action:
{action}

Generate JSON with two fields:
prompt: one short English sentence, no more than 28 words.
negative_prompt: comma-separated English keywords, no more than 24 keywords.
"""


def _fallback():
    return {
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
    }


def _clean_prompt(value: str, fallback: str, max_len: int = 300) -> str:
    value = (value or "").strip().replace("\n", " ")
    if not value:
        return fallback
    return value[:max_len]


def _clean_negative_prompt(value: str) -> str:
    value = _clean_prompt(value, DEFAULT_NEGATIVE_PROMPT, max_len=500)
    keywords = [item.strip() for item in value.split(",") if item.strip()]
    if not keywords:
        return DEFAULT_NEGATIVE_PROMPT
    return ", ".join(keywords[:24])


def generate_echomimicv3_prompts(
    avatar_name: str,
    avatar_description: str,
    speech_text: str,
    scene: str = "",
    action: str = "",
) -> dict:
    start = time.perf_counter()
    try:
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY is not set; using fallback prompts.")
            return _fallback()

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(
            avatar_description=avatar_description or avatar_name or "A realistic digital human",
            speech_text=speech_text or "",
            scene=scene or "A stable talking portrait shot",
            action=action or "speaking naturally",
        )
        completion = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        data = json.loads(content)
        prompt = _clean_prompt(data.get("prompt"), DEFAULT_PROMPT)
        negative_prompt = _clean_negative_prompt(data.get("negative_prompt"))
        logger.info(
            "EchoMimicV3 DeepSeek prompt generated in %.2fs: prompt=%s negative=%s",
            time.perf_counter() - start,
            prompt,
            negative_prompt,
        )
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
    except Exception:
        logger.exception("EchoMimicV3 DeepSeek prompt generation failed; using fallback prompts")
        return _fallback()
