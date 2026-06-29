import argparse
import time

from llm import llm_response


class DummyAvatarSession:
    def __init__(self):
        self.segments = []

    def put_msg_txt(self, msg, datainfo=None):
        self.segments.append(msg)
        print(f"[segment {len(self.segments)}] {msg}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Test DeepSeek LLM through LiveTalking llm.py")
    parser.add_argument(
        "message",
        nargs="?",
        default="请用一句话介绍你自己，并说明当前大模型连接正常。",
        help="message sent to the LLM",
    )
    args = parser.parse_args()

    avatar_session = DummyAvatarSession()
    start = time.perf_counter()
    llm_response(args.message, avatar_session, datainfo={"test": True})
    elapsed = time.perf_counter() - start

    print("\n========== LLM TEST RESULT ==========")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"segments: {len(avatar_session.segments)}")
    print("full_response:")
    print("".join(avatar_session.segments))
    print("=====================================")


if __name__ == "__main__":
    main()
