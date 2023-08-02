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

OPENAI_MODEL = "gpt-3.5-turbo"
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
    return openai_call_chat(
       messages=[
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": prompt}], *args, **kwargs
    )

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


def parse_json_in_str(s: str):
    s = f"-{s}-"
    if not len(s.split("```")) == 3:
        print_color(f"Error during parsing: {s}", Colors.GREY)
    else:
        return json.loads(s.split("```")[1])