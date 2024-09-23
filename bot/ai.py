import abc
import base64
from io import BytesIO
import logging
from typing import List, Tuple, Optional

import anthropic
import openai
import tiktoken

# Assume config is imported and contains necessary API keys and settings
import config

logger = logging.getLogger(__name__)

DEFAULT_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


class AIService(abc.ABC):
    @abc.abstractmethod
    async def send_message(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ) -> Tuple[str, Tuple[int, int], int]:
        pass

    @abc.abstractmethod
    async def send_message_stream(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ):
        pass

    @abc.abstractmethod
    async def send_vision_message(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ) -> Tuple[str, Tuple[int, int], int]:
        pass

    @abc.abstractmethod
    async def send_vision_message_stream(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ):
        pass


class OpenAIService(AIService):
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        openai.api_key = config.openai_api_key
        if config.openai_api_base:
            openai.api_base = config.openai_api_base
        self.buffer = None

    def set_buffer(self, buffer: BytesIO):
        self.buffer = buffer

    async def send_message(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ) -> Tuple[str, Tuple[int, int], int]:
        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            request_timeout=60.0,
        )
        answer = response.choices[0].message["content"].strip()
        n_input_tokens, n_output_tokens = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return answer, (n_input_tokens, n_output_tokens), 0

    async def send_message_stream(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ):
        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
        async for chunk in await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            request_timeout=60.0,
        ):
            if "content" in chunk.choices[0].delta:
                yield chunk.choices[0].delta.content

    async def send_vision_message(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ) -> Tuple[str, Tuple[int, int], int]:
        if self.model != "gpt-4-vision-preview":
            raise ValueError("Vision tasks require GPT-4 Vision model")

        messages = self._generate_prompt_messages(
            message, dialog_messages, chat_mode, image_buffer
        )
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            request_timeout=60.0,
        )
        answer = response.choices[0].message.content.strip()
        n_input_tokens, n_output_tokens = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        return answer, (n_input_tokens, n_output_tokens), 0

    async def send_vision_message_stream(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ):
        if self.model != "gpt-4-vision-preview":
            raise ValueError("Vision tasks require GPT-4 Vision model")

        messages = self._generate_prompt_messages(
            message, dialog_messages, chat_mode, image_buffer
        )
        async for chunk in await openai.ChatCompletion.acreate(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            request_timeout=60.0,
        ):
            if "content" in chunk.choices[0].delta:
                yield chunk.choices[0].delta.content

    def _generate_prompt_messages(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ) -> List[dict]:
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = [{"role": "system", "content": prompt}]

        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message.get("user")})
            messages.append({"role": "assistant", "content": dialog_message.get("bot")})

        if image_buffer:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": message},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_buffer.getvalue()).decode('utf-8')}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": message})

        return messages


class ClaudeService(AIService):
    def __init__(self, model="claude-3-sonnet-20240229"):
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.buffer = None

    def set_buffer(self, buffer: BytesIO):
        if buffer:
            self.buffer = buffer

    async def send_message(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ) -> Tuple[str, Tuple[int, int], int]:
        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_COMPLETION_OPTIONS["max_tokens"],
            temperature=DEFAULT_COMPLETION_OPTIONS["temperature"],
        )
        answer = response.content[0].text.strip()
        n_input_tokens, n_output_tokens = (
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return answer, (n_input_tokens, n_output_tokens), 0

    async def send_message_stream(
        self, message: str, dialog_messages: List[dict], chat_mode: str
    ):
        messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
        async with self.client.messages.stream(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_COMPLETION_OPTIONS["max_tokens"],
            temperature=DEFAULT_COMPLETION_OPTIONS["temperature"],
        ) as stream:
            answer = ""
            async for event in stream:
                if event.type == "content_block_delta":
                    answer += event.delta.text
                    yield "not_finished", answer, (0, 0), 0

            yield "finished", answer, (0, 0), 0

    async def send_vision_message(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ) -> Tuple[str, Tuple[int, int], int]:
        messages = self._generate_prompt_messages(
            message, dialog_messages, chat_mode, image_buffer
        )
        response = await self.client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_COMPLETION_OPTIONS["max_tokens"],
            temperature=DEFAULT_COMPLETION_OPTIONS["temperature"],
        )
        answer = response.content[0].text.strip()
        n_input_tokens, n_output_tokens = (
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return answer, (n_input_tokens, n_output_tokens), 0

    async def send_vision_message_stream(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ):
        messages = self._generate_prompt_messages(
            message, dialog_messages, chat_mode, image_buffer
        )
        async with self.client.messages.stream(
            model=self.model,
            messages=messages,
            max_tokens=DEFAULT_COMPLETION_OPTIONS["max_tokens"],
            temperature=DEFAULT_COMPLETION_OPTIONS["temperature"],
        ) as stream:
            answer = ""
            async for response in stream:
                if response.type == "content_block_delta":
                    answer += response.delta.text
                    yield "not_finished", answer, (0, 0), 0

            yield "finished", answer, (0, 0), 0

    def _generate_prompt_messages(
        self,
        message: str,
        dialog_messages: List[dict],
        chat_mode: str,
        image_buffer: Optional[BytesIO] = None,
    ):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        messages = []

        # print(dialog_messages)

        for dialog_message in dialog_messages:
            if dialog_message.get("user"):
                messages.append({"role": "user", "content": dialog_message.get("user")})
            if dialog_message.get("bot"):
                messages.append(
                    {"role": "assistant", "content": dialog_message.get("bot")}
                )

        if image_buffer:
            image_content = anthropic.ImageContent(
                type="image",
                source=anthropic.ImageSource(
                    type="base64",
                    media_type="image/jpeg",
                    data=base64.b64encode(image_buffer.getvalue()).decode("utf-8"),
                ),
            )
            messages.append({"role": "user", "content": [message, image_content]})
        elif self.buffer:
            content = self.buffer.read().decode("utf-8")
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": f"{message}{content}"}]}
            )
            self.buffer = None
        else:
            messages.append({"role": "user", "content": message})

        # messages.append({"role": "assistant", "content": prompt.strip()})

        logging.warning(messages)

        return messages


class AIFactory:
    @staticmethod
    def create(service: str = "openai", model: str = "gpt-4o-mini"):
        if service == "openai":
            return OpenAIService(model)
        elif service == "claude":
            return ClaudeService(model)
        else:
            raise ValueError(f"Unknown service: {service}")


# Usage example:
async def sample_main():
    # adapter = AIFactory.create(service="openai", model="gpt-3.5-turbo")
    # To switch to Claude:
    adapter = AIFactory.create(service="claude", model="claude-3-sonnet-20240229")

    message = "Hello, how are you?"
    dialog_messages = []
    chat_mode = "assistant"

    # response, tokens, _ = await adapter.send_message(
    #     message, dialog_messages, chat_mode
    # )
    # print(f"Response: {response}")
    # print(f"Tokens used: {tokens}")

    print("Streaming response:")
    async for chunk in adapter.send_message_stream(message, dialog_messages, chat_mode):
        print(chunk, end="", flush=True)
    print()



# import asyncio
# asyncio.run(sample_main())