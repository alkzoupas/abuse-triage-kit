"""LLM reviewer: render an account timeline into a prompt, ask a provider, parse a verdict.

Providers: MockLLM (default, offline, a hash of the prompt) and AnthropicProvider
(optional, real model, used only when the owner runs it with their own key).
"""
