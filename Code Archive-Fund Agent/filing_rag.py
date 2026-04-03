#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:30:44 2026

@author: fabriziocoiai
"""
# filing_rag.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
import numpy as np
import pandas as pd
import re


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...


@dataclass
class RagChunk:
    chunk_id: int
    doc_id: str
    ticker: str
    filed: pd.Timestamp
    source: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


def chunk_text(text: str, chunk_chars: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + chunk_chars, n)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out


class FilingRAG:
    def __init__(self):
        self.chunks: List[RagChunk] = []
        self._next_id = 0

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return num / den

    def add_document(
        self,
        embedder: Embedder,
        *,
        doc_id: str,
        ticker: str,
        filed: pd.Timestamp,
        source: str,
        text: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        metadata = metadata or {}
        pieces = chunk_text(text)
        if not pieces:
            return
        embs = embedder.embed(pieces)
        for i, piece in enumerate(pieces):
            self.chunks.append(
                RagChunk(
                    chunk_id=self._next_id,
                    doc_id=doc_id,
                    ticker=ticker,
                    filed=pd.Timestamp(filed),
                    source=source,
                    text=piece,
                    metadata={**metadata, "chunk_index": str(i)},
                    embedding=embs[i],
                )
            )
            self._next_id += 1

    def retrieve(
        self,
        embedder: Embedder,
        *,
        ticker: str,
        query: str,
        asof: pd.Timestamp,
        top_k: int = 5,
    ) -> List[RagChunk]:
        qemb = embedder.embed([query])[0]
        candidates = [
            c for c in self.chunks
            if c.ticker == ticker and c.filed <= pd.Timestamp(asof)
        ]
        scored = []
        for c in candidates:
            if c.embedding is None:
                continue
            scored.append((self._cos(qemb, c.embedding), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]