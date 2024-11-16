from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


from PIL import Image


@dataclass
class TextChatMessageContent:
    text: str


@dataclass
class ImageChatMessageContent:
    image: str


@dataclass
class ChatMessage:
    role: str
    content: Union[str, List[Union[TextChatMessageContent, ImageChatMessageContent]]]

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            text_item_count = 0
            for content_item in self.content:
                if isinstance(content_item, TextChatMessageContent):
                    text_item_count += 1

            if text_item_count != 1:
                raise ValueError("A chat message must have exactly one text content member")

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content

        for content_item in self.content:
            if isinstance(content_item, TextChatMessageContent):
                return content_item.text

        raise ValueError("No text found for this message")

    @text.setter
    def text(self, new_value: str) -> None:
        if isinstance(self.content, str):
            self.content = new_value
        else:
            for content_item in self.content:
                if isinstance(content_item, TextChatMessageContent):
                    content_item.text = new_value
                    break

    def copy(self) -> "ChatMessage":
        return ChatMessage.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        if isinstance(self.content, str):
            return dict(role=self.role, content=self.content)

        content_dicts: List[Dict[str, Any]] = []
        for content_item in self.content:
            if isinstance(content_item, TextChatMessageContent):
                content_dicts.append(dict(type="text", text=content_item.text))
            else:
                content_dicts.append(dict(type="image_url", image_url=dict(url=content_item.image)))

        return dict(role=self.role, content=content_dicts)

    @staticmethod
    def from_text(text: str) -> "ChatMessage":
        return ChatMessage(role="user", content=text)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ChatMessage":
        role = d["role"]
        content = d["content"]

        if isinstance(content, str):
            return ChatMessage(role=role, content=content)

        final_content: List[Union[TextChatMessageContent, ImageChatMessageContent]] = []
        for content_item in content:
            _type = content_item["type"]
            if _type == "text":
                final_content.append(TextChatMessageContent(content_item["text"]))
            elif _type == "image_url":
                # TODO: validate b64 encoding?
                final_content.append(ImageChatMessageContent(content_item["image_url"]["url"]))
            else:
                raise ValueError(f"Unexpected message content type {_type}")

        return ChatMessage(role=role, content=final_content)


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: List[int]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None


@dataclass
class RawInputItem:
    index: int
    messages: List[ChatMessage]

    @staticmethod
    def from_text(index: int, text: str) -> "RawInputItem":
        return RawInputItem(index=index, messages=[ChatMessage.from_text(text)])

    @staticmethod
    def from_message_dicts(index: int, message_dicts: List[Dict[str, Any]]) -> "RawInputItem":
        messages = [ChatMessage.from_dict(m_dict) for m_dict in message_dicts]
        return RawInputItem(index=index, messages=messages)


@dataclass
class PreparedInputItem:
    index: int
    token_ids: List[int]
    image_data: Optional[List[Image.Image]] = None


@dataclass
class TextItem:
    index: int
    text: str


@dataclass
class TokenizedItem:
    index: int
    token_ids: List[int]


class CompletionError(str, Enum):
    CONTEXT_TOO_LONG = "CONTEXT_TOO_LONG"


@dataclass
class CompletedItem:
    index: int
    outputs: List[CompletionOutput]
    error: Optional[CompletionError] = None


@dataclass
class Message:
    message_id: str
    receipt_handle: str
    bucket_name: str
    object_key: str
