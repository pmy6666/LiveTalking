import os
import time
from typing import TYPE_CHECKING

from utils.logger import logger

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict = {}):
    try:
        start = time.perf_counter()
        from openai import OpenAI

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY is not set; skip LLM response.")
            return

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")

        completion = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            messages=[
                {
                    "role": "system",
                    "content": "你是一个简洁、自然、适合口语播报的中文数字人助手，请直接回答用户问题。",
                },
                {"role": "user", "content": message},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )

        result = ""
        first = True
        for chunk in completion:
            if len(chunk.choices) > 0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False

                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue

                lastpos = 0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：?":
                        result = result + msg[lastpos : i + 1]
                        lastpos = i + 1
                        if len(result) > 10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result, datainfo)
                            result = ""
                result = result + msg[lastpos:]

        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result, datainfo)

    except Exception:
        logger.exception("llm exception:")
        return
