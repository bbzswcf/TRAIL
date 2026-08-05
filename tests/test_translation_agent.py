import json
from types import SimpleNamespace
import unittest
from unittest import mock

import requests

from c2rust_translator.translation_agent import Agent, create_agent


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class RateLimitedResponse:
    status_code = 429

    def __init__(self, retry_after=None):
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = retry_after

    def raise_for_status(self):
        raise requests.HTTPError("429 Client Error: Too Many Requests", response=self)


class ClientErrorResponse:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.payload)


class FlakySession:
    def __init__(self, failures, payload):
        self.failures = failures
        self.payload = payload
        self.calls = 0

    def post(self, url, *, headers, json, timeout):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"temporary failure {self.calls}")
        return FakeResponse(self.payload)


class RateLimitedSession:
    def __init__(self, retry_after=None):
        self.retry_after = retry_after
        self.calls = 0

    def post(self, url, *, headers, json, timeout):
        self.calls += 1
        if self.calls == 1:
            return RateLimitedResponse(self.retry_after)
        return FakeResponse({"output_text": "```rust\nfn translated() {}\n```"})


class ClientErrorSession:
    def __init__(self, status_code, failures=None):
        self.status_code = status_code
        self.calls = 0
        self.failures = failures

    def post(self, url, *, headers, json, timeout):
        self.calls += 1
        if self.failures is not None and self.calls > self.failures:
            return FakeResponse({"output_text": "```rust\nfn translated() {}\n```"})
        return ClientErrorResponse(self.status_code)


class FakeChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="```rust\nfn chat_translated() {}\n```"))],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=17, total_tokens=30),
        )


class BadRequestChatCompletions:
    def __init__(self, message):
        self.message = message
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        exc = RuntimeError(self.message)
        exc.status_code = 400
        raise exc


class BadRequestChatClient:
    def __init__(self, message):
        self.completions = BadRequestChatCompletions(message)


class FakeChatClient:
    def __init__(self):
        self.completions = FakeChatCompletions()


class FakeGoogleModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            text="```rust\nfn google_translated() {}\n```",
            usage_metadata=SimpleNamespace(
                prompt_token_count=19,
                candidates_token_count=23,
                total_token_count=42,
            ),
        )


class FakeGoogleClient:
    def __init__(self):
        self.models = FakeGoogleModels()


class ResponsesAgentTest(unittest.TestCase):
    def test_openai_responses_protocol_sends_required_headers_and_extracts_text(self):
        session = FakeSession(
            {
                "output_text": "```rust\nfn translated() {}\n```",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                },
            }
        )
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            http_client=session,
        )

        result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn translated() {}")
        self.assertEqual(result.usage.prompt_tokens, 11)
        self.assertEqual(result.usage.completion_tokens, 7)
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(len(session.calls), 1)

        call = session.calls[0]
        self.assertEqual(call["url"], "https://example.test/responses")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(call["headers"]["X-TT-logid"], "log-456")
        self.assertEqual(json.loads(call["headers"]["extra"]), {"session_id": "session-123"})
        self.assertEqual(call["json"]["model"], "gpt-5.4-mini-2026-03-17")
        self.assertEqual(call["json"]["instructions"], "system prompt")
        self.assertEqual(call["json"]["input"], "user prompt")
        self.assertEqual(call["json"]["temperature"], 1.0)
        self.assertEqual(call["json"]["reasoning"], {"effort": "medium"})
        self.assertEqual(call["json"]["include"], ["reasoning.encrypted_content"])

    def test_openai_responses_protocol_extracts_message_content_text(self):
        session = FakeSession(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "```rust\nfn from_content() {}\n```"}
                        ],
                    }
                ]
            }
        )
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            session_id="session-123",
            logid="log-456",
            http_client=session,
        )

        result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn from_content() {}")

    def test_openai_responses_retry_uses_four_second_exponential_backoff(self):
        session = FlakySession(
            2,
            {
                "output_text": "```rust\nfn translated() {}\n```",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            },
        )
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            retries=3,
            retry_initial_delay=4,
            retry_max_delay=60,
            http_client=session,
        )

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn translated() {}")
        self.assertEqual(session.calls, 3)
        sleep.assert_has_calls([mock.call(4), mock.call(8)])

    def test_retry_backoff_is_capped_at_configured_max_delay(self):
        session = FlakySession(3, {"output_text": "```rust\nfn translated() {}\n```"})
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            retries=4,
            retry_initial_delay=40,
            retry_max_delay=60,
            http_client=session,
        )

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            agent.translate("system prompt", "user prompt")

        sleep.assert_has_calls([mock.call(40), mock.call(60), mock.call(60)])

    def test_rate_limit_response_uses_longer_retry_delay(self):
        session = RateLimitedSession()
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            retries=2,
            retry_initial_delay=4,
            retry_max_delay=60,
            rate_limit_retry_delay=90,
            http_client=session,
        )

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn translated() {}")
        sleep.assert_called_once_with(90)

    def test_rate_limit_response_respects_retry_after_header(self):
        session = RateLimitedSession(retry_after="75")
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            retries=2,
            retry_initial_delay=4,
            retry_max_delay=60,
            rate_limit_retry_delay=60,
            http_client=session,
        )

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            agent.translate("system prompt", "user prompt")

        sleep.assert_called_once_with(75.0)

    def test_bad_request_response_is_retried(self):
        session = ClientErrorSession(400, failures=1)
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            retries=2,
            retry_initial_delay=4,
            http_client=session,
        )

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn translated() {}")
        self.assertEqual(session.calls, 2)
        sleep.assert_called_once_with(4)

    def test_non_retryable_http_errors_are_not_retried(self):
        for status_code in (401, 403, 404):
            with self.subTest(status_code=status_code):
                session = ClientErrorSession(status_code)
                agent = Agent(
                    api_key="secret",
                    api_url="https://example.test/responses",
                    base_url="https://unused.test",
                    model="gpt-5.4-mini-2026-03-17",
                    protocol="openai-responses",
                    retries=20,
                    http_client=session,
                )

                with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
                    with self.assertRaises(RuntimeError):
                        agent.translate("system prompt", "user prompt")

                self.assertEqual(session.calls, 1)
                sleep.assert_not_called()

    def test_permanent_bad_request_size_errors_are_not_retried_for_chat(self):
        agent = Agent(
            api_key="secret",
            base_url="https://example.test/chat",
            model="kimi-k2.5",
            protocol="openai-chat",
            retries=20,
        )
        fake_chat = BadRequestChatClient(
            "400:Invalid request: Your request exceeded model token limit: 262144"
        )
        agent.client = SimpleNamespace(chat=fake_chat)

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "non-retryable HTTP error"):
                agent.translate("system prompt", "user prompt")

        self.assertEqual(fake_chat.completions.calls, 1)
        sleep.assert_not_called()

    def test_total_message_size_bad_request_is_not_retried_for_chat(self):
        agent = Agent(
            api_key="secret",
            base_url="https://example.test/chat",
            model="kimi-k2.5",
            protocol="openai-chat",
            retries=20,
        )
        fake_chat = BadRequestChatClient(
            "400:Invalid request: total message size 10422423 exceeds limit 4194304"
        )
        agent.client = SimpleNamespace(chat=fake_chat)

        with mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "non-retryable HTTP error"):
                agent.translate("system prompt", "user prompt")

        self.assertEqual(fake_chat.completions.calls, 1)
        sleep.assert_not_called()

    def test_min_request_interval_paces_sequential_responses_calls(self):
        session = FakeSession({"output_text": "```rust\nfn translated() {}\n```"})
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/responses",
            base_url="https://unused.test",
            model="gpt-5.4-mini-2026-03-17",
            protocol="openai-responses",
            min_request_interval=10,
            http_client=session,
        )

        with (
            mock.patch("c2rust_translator.translation_agent.time.monotonic", side_effect=[100.0, 105.0]),
            mock.patch("c2rust_translator.translation_agent.time.sleep") as sleep,
        ):
            agent.translate("system prompt", "first prompt")
            agent.translate("system prompt", "second prompt")

        self.assertEqual(len(session.calls), 2)
        sleep.assert_called_once_with(5.0)

    def test_chat_protocol_can_send_enable_thinking_with_max_tokens(self):
        agent = Agent(
            api_key="secret",
            base_url="https://example.test/chat",
            model="ali-deepseek-v4-flash",
            protocol="openai-chat",
            max_tokens=128000,
            chat_extra_body={"enable_thinking": True},
            chat_token_parameter="max_tokens",
        )
        fake_client = SimpleNamespace(chat=FakeChatClient())
        agent.client = fake_client

        result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn chat_translated() {}")
        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["model"], "ali-deepseek-v4-flash")
        self.assertEqual(call["max_tokens"], 128000)
        self.assertNotIn("max_completion_tokens", call)
        self.assertEqual(call["extra_body"], {"enable_thinking": True})

    def test_chat_protocol_can_send_max_completion_tokens_without_extra_body(self):
        agent = Agent(
            api_key="secret",
            base_url="https://example.test/chat",
            model="kimi-k2.5",
            protocol="openai-chat",
            max_tokens=128000,
            chat_extra_body=None,
            chat_token_parameter="max_completion_tokens",
        )
        fake_client = SimpleNamespace(chat=FakeChatClient())
        agent.client = fake_client

        agent.translate("system prompt", "user prompt")

        call = fake_client.chat.completions.calls[0]
        self.assertEqual(call["model"], "kimi-k2.5")
        self.assertEqual(call["max_completion_tokens"], 128000)
        self.assertNotIn("max_tokens", call)
        self.assertNotIn("extra_body", call)

    def test_google_genai_protocol_uses_native_generate_content_payload(self):
        fake_client = FakeGoogleClient()
        agent = Agent(
            api_key="secret",
            api_url="https://example.test/google/",
            model="gemini-3.1-p-priority",
            protocol="google-genai",
            max_tokens=65536,
            google_client=fake_client,
        )

        result = agent.translate("system prompt", "user prompt")

        self.assertEqual(result.code, "fn google_translated() {}")
        self.assertEqual(result.usage.prompt_tokens, 19)
        self.assertEqual(result.usage.completion_tokens, 23)
        self.assertEqual(result.usage.total_tokens, 42)
        call = fake_client.models.calls[0]
        self.assertEqual(call["model"], "gemini-3.1-p-priority")
        self.assertEqual(call["contents"], "user prompt")
        self.assertEqual(call["config"]["system_instruction"], "system prompt")
        self.assertEqual(call["config"]["temperature"], 1.0)
        self.assertEqual(call["config"]["max_output_tokens"], 65536)

    def test_create_agent_reads_retry_policy_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
                "max_retries": 20,
                "retry_initial_delay": 4,
                "retry_max_delay": 60,
            }
        )

        self.assertEqual(agent.retries, 20)
        self.assertEqual(agent.retry_initial_delay, 4)
        self.assertEqual(agent.retry_max_delay, 60)

    def test_create_agent_reads_rate_limit_retry_delay_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
                "rate_limit_retry_delay": 120,
            }
        )

        self.assertEqual(agent.rate_limit_retry_delay, 120)

    def test_create_agent_reads_timeout_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
                "timeout": 900,
            }
        )

        self.assertEqual(agent.timeout, 900)

    def test_create_agent_reads_min_request_interval_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
                "min_request_interval": 7.5,
            }
        )

        self.assertEqual(agent.min_request_interval, 7.5)

    def test_create_agent_defaults_to_large_output_token_budget(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
            }
        )

        self.assertEqual(agent.max_tokens, 128000)

    def test_create_agent_reads_reasoning_payload_options_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gpt-5.4-mini-2026-03-17",
                "protocol": "openai-responses",
                "api_url": "https://example.test/responses",
                "temperature": 1.0,
                "reasoning_effort": "medium",
                "include": ["reasoning.encrypted_content"],
            }
        )

        self.assertEqual(agent.temperature, 1.0)
        self.assertEqual(agent.reasoning_effort, "medium")
        self.assertEqual(agent.include, ["reasoning.encrypted_content"])

    def test_create_agent_reads_chat_completion_payload_options_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "kimi-k2.5",
                "protocol": "openai-chat",
                "base_url": "https://example.test/chat",
                "max_tokens": 128000,
                "chat_extra_body": None,
                "chat_token_parameter": "max_completion_tokens",
            }
        )

        self.assertEqual(agent.max_tokens, 128000)
        self.assertIsNone(agent.chat_extra_body)
        self.assertEqual(agent.chat_token_parameter, "max_completion_tokens")

    def test_create_agent_reads_google_genai_payload_options_from_config(self):
        agent = create_agent(
            {
                "api_key": "secret",
                "model_name": "gemini-3.1-p-priority",
                "protocol": "google-genai",
                "api_url": "https://example.test/google/",
                "max_tokens": 65536,
                "google_api_version": "v1",
            }
        )

        self.assertEqual(agent.protocol, "google-genai")
        self.assertEqual(agent.api_url, "https://example.test/google/")
        self.assertEqual(agent.max_tokens, 65536)
        self.assertEqual(agent.google_api_version, "v1")

    def test_google_genai_client_uses_millisecond_timeout_and_configured_api_version(self):
        with mock.patch("c2rust_translator.translation_agent.genai.Client") as client:
            Agent(
                api_key="secret",
                api_url="https://example.test/google/",
                model="gemini-3.1-p-priority",
                protocol="google-genai",
                google_api_version="v1",
                timeout=300,
            )

        http_options = client.call_args.kwargs["http_options"]
        self.assertEqual(http_options.base_url, "https://example.test/google/")
        self.assertEqual(http_options.api_version, "v1")
        self.assertEqual(http_options.timeout, 300000)

    def test_openai_chat_client_disables_sdk_retries(self):
        with mock.patch("c2rust_translator.translation_agent.OpenAI") as client:
            Agent(
                api_key="secret",
                base_url="https://example.test/chat",
                model="kimi-k2.5",
                protocol="openai-chat",
                timeout=300,
            )

        self.assertEqual(client.call_args.kwargs["timeout"], 300)
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)


if __name__ == "__main__":
    unittest.main()
