#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:22:43 2026

@author: fabriziocoiai
"""
# llm_backend.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Protocol

import numpy as np
try:
    from google import genai
    from google.genai import errors
except Exception:  # pragma: no cover - optional dependency path
    genai = None
    errors = None
from openai import OpenAI


class LLMBackend(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        ...

    def embed(self, texts: List[str]) -> np.ndarray:
        ...


class GeminiBackend:
    def __init__(
        self,
        chat_model: str = "gemini-2.0-flash",
        embed_model: str = "gemini-embedding-001",
        api_key_env: str = "GEMINI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ):
        if genai is None:
            raise RuntimeError("google-genai package is not available for GeminiBackend.")
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set. Export your Gemini API key first.")

        self.client = genai.Client(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @staticmethod
    def _render_messages(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "user").upper()
            content = m.get("content") or ""
            parts.append(f"{role}:\n{content}")
        return "\n\n".join(parts)

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt = self._render_messages(messages)

        for attempt in range(5):
            try:
                resp = self.client.models.generate_content(
                    model=self.chat_model,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                    },
                )
                text = getattr(resp, "text", None) or str(resp)
                return {"content": text.strip()}
            except errors.ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        raise RuntimeError("Gemini chat failed after retries due to rate limits/quota.")

    # def embed(self, texts: List[str]) -> np.ndarray:
    #     vectors: List[List[float]] = []

    #     for t in texts:
    #         try:
    #             r = self.client.models.embed_content(
    #                 model=self.embed_model,
    #                 contents=t,
    #             )
    #         except errors.ClientError as e:
    #             raise RuntimeError(
    #                 f"Embedding failed for model '{self.embed_model}'. "
    #                 f"Original error: {e}"
    #             ) from e

            # emb = getattr(r, "embedding", None)
            # if emb is None or getattr(emb, "values", None) is None:
            #     raise RuntimeError("Unexpected embed response format from Gemini SDK.")
            # vectors.append(list(emb.values))
            
        # return np.array(vectors, dtype=np.float32)
    
    
    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
    
        for t in texts:
            try:
                r = self.client.models.embed_content(
                    model=self.embed_model,
                    contents=t,
                )
            except errors.ClientError as e:
                raise RuntimeError(
                    f"Embedding failed for model '{self.embed_model}'. "
                    f"Original error: {e}"
                ) from e
    
            # Gemini SDK returns `embeddings` in the documented examples.
            embs = getattr(r, "embeddings", None)
    
            if embs is None:
                # fallback for SDK shape variations
                single = getattr(r, "embedding", None)
                if single is not None and getattr(single, "values", None) is not None:
                    vectors.append(list(single.values))
                    continue
    
                # helpful debug so we can inspect the actual object if needed
                raise RuntimeError(
                    f"Unexpected embed response format from Gemini SDK. "
                    f"Response type={type(r)} repr={r}"
                )
    
            if len(embs) == 0:
                raise RuntimeError("Gemini embed_content returned no embeddings.")
    
            first = embs[0]
            values = getattr(first, "values", None)
    
            if values is None:
                raise RuntimeError(
                    f"Gemini embed_content returned embeddings without `.values`. "
                    f"First item type={type(first)} repr={first}"
                )
    
            vectors.append(list(values))
    
        return np.array(vectors, dtype=np.float32)


class OpenAIBackend:
    def __init__(
        self,
        chat_model: str = "gpt-4.1-mini",
        embed_model: str = "text-embedding-3-small",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
        max_output_tokens: int = 800,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set. Export your OpenAI API key first.")

        self.client = OpenAI(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                return {"content": text}
            except Exception as e:
                emsg = str(e).lower()
                if "rate limit" in emsg or "429" in emsg:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        raise RuntimeError("OpenAI chat failed after retries due to rate limits/quota.")

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors: List[List[float]] = []
        for text in texts:
            resp = self.client.embeddings.create(
                model=self.embed_model,
                input=text,
            )
            vectors.append(list(resp.data[0].embedding))
        return np.array(vectors, dtype=np.float32)
        
    
    
    
    