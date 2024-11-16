import unittest

from birr.batch_inference.data_models import ChatMessage, ImageChatMessageContent, TextChatMessageContent


class TestDataModels(unittest.TestCase):
    def test__chat_message_init_fails_if_no_text_in_content(self):
        content = [ImageChatMessageContent("base64asdfaswe")]

        with self.assertRaises(ValueError):
            ChatMessage(role="user", content=content)

    def test__chat_message_init_fails_if_more_than_one_text_in_content(self):
        content = [
            ImageChatMessageContent("base64asdf"),
            TextChatMessageContent("asdf"),
            TextChatMessageContent("fdsa"),
        ]

        with self.assertRaises(ValueError):
            ChatMessage(role="user", content=content)

    def test__chat_message_text_accessors_work(self):
        list_content = [ImageChatMessageContent("base64asdf"), TextChatMessageContent("asdf")]

        chat_message1 = ChatMessage(role="user", content=list_content)
        self.assertEqual(chat_message1.text, "asdf")
        chat_message1.text = "fdsa"
        self.assertEqual(chat_message1.text, "fdsa")

        chat_message2 = ChatMessage(role="user", content="asdf")
        self.assertEqual(chat_message2.text, "asdf")
        chat_message2.text = "fdsa"
        self.assertEqual(chat_message2.text, "fdsa")

    def test__to_dict_serializes_to_openai_format(self):
        message1 = ChatMessage(role="user", content="asdf")
        self.assertEqual({"role": "user", "content": "asdf"}, message1.to_dict())

        message2 = ChatMessage(
            role="user", content=[ImageChatMessageContent("base64asdf"), TextChatMessageContent("asdf")]
        )
        self.assertEqual(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "base64asdf"}},
                    {"type": "text", "text": "asdf"},
                ],
            },
            message2.to_dict(),
        )

    def test__from_text(self):
        message = ChatMessage.from_text("asdf")
        self.assertEqual(ChatMessage(role="user", content="asdf"), message)

    def test__from_dict(self):
        message = ChatMessage.from_dict(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "base64asdf"}},
                    {"type": "text", "text": "asdf"},
                ],
            }
        )
        expected_message = ChatMessage(
            role="user", content=[ImageChatMessageContent("base64asdf"), TextChatMessageContent("asdf")]
        )

        self.assertEqual(expected_message, message)
