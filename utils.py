""" Module providing for OpenAI"""
import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import openai
from openai.error import (APIConnectionError, APIError, RateLimitError,
                          ServiceUnavailableError, Timeout)
from retrying import retry

OPENAI_MODEL = "gpt-4"
OPENAI_RETRYABLE_ERRORS = (
    APIError,
    Timeout,
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError,
)


# Color enum
class Colors(Enum):
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"
    RESET = "\033[0m"


OPENAI_MODEL_TYPE = {
    "text-davinci-003": "completion",
    "gpt-3.5-turbo": "chat",
    "gpt-4": "chat",
}


# Function to print text with color
def print_color(text, color, *args, **kwargs):
    print(f"{color.value}{text}{Colors.RESET.value}", *args, **kwargs)


def stream_and_print(f: callable, color: Colors = Colors.RESET):
    text = ""
    for part in f():
        text += part
        print_color(part, color=color, end="")
    return text


def gwithlen(g, len):
    for idx, out in enumerate(g):
        if idx == len:
            break
        yield out


def yymmdd():
    today = datetime.now().date()
    return today.strftime("%y%m%d")


@retry(
    stop_max_attempt_number=99999,
    retry_on_exception=lambda e: isinstance(e, OPENAI_RETRYABLE_ERRORS),
)
def openai_call_completion(
    prompt: str,
    model: str = "text-davinci-003",
    stream=False,
    call_args: Dict = {},
) -> Union[Tuple[None, None, None], Tuple[str, bool, Any]]:
    def _get_text(r, stream=False):
        return r["choices"][0].text

    def _get_text_stream(r):
        full_answer = ""
        for x in r:
            part = _get_text(x, stream=True)
            full_answer += part
            yield part

    time.time()

    default_args = dict(
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    default_args.update(call_args)
    response = openai.Completion.create(
        prompt=prompt,
        engine=model,
        stream=stream,
        **default_args,
    )

    if stream:
        return _get_text_stream(response)
    return _get_text(response)


def openai_call_chat_single(prompt, *args, **kwargs):
    # Try to load the cache file
    try:
        with open('openai_calls.json', 'r') as f:
            for line in f:
                cached = json.loads(line)
                # If the prompt matches, return the cached response
                if cached["prompt"] == prompt:
                    print("Found a cached response!")
                    return cached["response"]
    except FileNotFoundError:
        print("Cache file not found, creating a new one.")

    print_color(f"==OPENAI API CALL ====\n{prompt}\n===============", Colors.GREY)
    out = openai_call_chat(
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": prompt}], *args, **kwargs
    )
    print_color(f"##OPENAI API RESPONSE ####\n{out}\n################", Colors.GREY)
    with open('openai_calls.json', 'a') as f:
        f.write(json.dumps({"prompt": prompt, "response": out}))
        f.write("\n")

    return out

@retry(
    stop_max_attempt_number=99999,
    retry_on_exception=lambda e: isinstance(e, OPENAI_RETRYABLE_ERRORS),
)
def openai_call_chat(
    messages: List[dict],
    model: str = OPENAI_MODEL,
    stream=False,
    call_args: Dict = {},
) -> Union[Tuple[None, None, None], Tuple[str, bool, Any]]:
    def _get_text(r, stream=False):
        choice = r["choices"][0]
        if stream:
            part = (
                choice["delta"]["content"]
                if "content" in choice["delta"].keys()
                else ""
            )
        else:
            part = choice["message"]["content"]
        return part

    def _get_text_stream(r):
        full_answer = ""
        for x in r:
            part = _get_text(x, stream=True)
            full_answer += part
            yield part

    full_prompt = "\n".join([m["content"] for m in messages])
    json.dumps(
        full_prompt, ensure_ascii=False
    )  # to avoid special char spoil the log stream

    time.time()

    default_args = dict(
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    default_args.update(call_args)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        stream=stream,
        **default_args,
    )

    if stream:
        return _get_text_stream(response)
    return _get_text(response)


class Roles(Enum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2


class SimpleChatPromptTemplate:
    def __init__(self, prompt: list[tuple[str, str]]):
        self.prompt = prompt

    def format(self, *args, **kwargs):
        p = []
        for role, content in self.prompt:
            p.append({"role": role, "content": content.format(*args, **kwargs)})
        return p

    def __str__(self) -> str:
        return "\n\n".join([f"{r}: {c}" for r, c in self.prompt])


def parse_json_in_str(s: str, default: Any = []):
    s = f"-{s}-"
    try:
        if not len(s.split("```")) == 3:
            print_color(f"Error during parsing: {s}", Colors.GREY)
            return default
        else:
            return json.loads(s.split("```")[1])
    except Exception as e:
        print_color(f"Error during parsing: {s}", Colors.GREY)
        return default
