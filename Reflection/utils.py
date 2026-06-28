from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set

from Reflection.types import FactCategory


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9_@:%/+.-]+|[\u4e00-\u9fff]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？；;])\s+|\n+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "this",
    "to",
    "user",
    "we",
    "with",
    "you",
    "your",
}


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def split_sentences(text: str) -> List[str]:
    """Break text into short fact candidates while preserving stable order."""

    parts = SENTENCE_SPLIT_RE.split(text)
    return [part.strip().strip(" .!?;。！？；") for part in parts if part.strip()]


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def tokenize(text: str) -> List[str]:
    cleaned_tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if "@" not in token:
            token = token.strip(".,;:!?()[]{}\"'")
        if token and token not in STOPWORDS:
            cleaned_tokens.append(token)
    return cleaned_tokens


def unique_tokens(text: str) -> Set[str]:
    return set(tokenize(text))


def lexical_support(query: str, document: str) -> float:
    """Coverage-style support score used by provenance matching."""

    query_tokens = unique_tokens(query)
    if not query_tokens:
        return 0.0
    document_tokens = unique_tokens(document)
    if not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / len(query_tokens)


def contains_email(text: str) -> bool:
    return EMAIL_RE.search(text) is not None


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0).lower() if match else ""


def has_any(text: str, phrases: Iterable[str]) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in phrases)


def classify_fact(text: str) -> FactCategory:
    """Heuristic categorization keeps the defense package dependency-free."""

    lower = text.lower()
    if has_any(lower, {"api key", "access token", "password", "secret", "credential"}):
        return FactCategory.CREDENTIAL
    if has_any(
        lower,
        {
            "remember this",
            "save this",
            "store this",
            "write this to memory",
            "for future conversations",
            "when summarizing",
            "future summaries should",
            "always mention",
            "must remember",
            "reflect that",
        },
    ):
        return FactCategory.INSTRUCTION
    if contains_email(text) or has_any(lower, {"contact", "mailbox", "email address", "reach me at"}):
        return FactCategory.CONTACT
    if has_any(lower, {"my name is", "call me", "i am ", "我是", "我叫"}):
        return FactCategory.IDENTITY
    if has_any(lower, {"i prefer", "favorite", "likes", "prefers", "enjoys", "喜欢"}):
        return FactCategory.PREFERENCE
    if has_any(lower, {"deadline", "task", "todo", "need to", "must finish", "plan to"}):
        return FactCategory.TASK
    return FactCategory.OTHER


def extract_slot_key(text: str, category: FactCategory) -> str:
    """Approximate field identity for contradiction checks."""

    lower = text.lower()
    if category == FactCategory.CONTACT and contains_email(text):
        return "contact_email"
    if category == FactCategory.IDENTITY and "name" in lower:
        return "name"
    if category == FactCategory.PREFERENCE:
        if "coffee" in lower or "tea" in lower:
            return "drink_preference"
        if "color" in lower or "colour" in lower:
            return "color_preference"
        return "general_preference"
    return f"{category.value}:{' '.join(tokenize(text)[:3])}"


def extract_canonical_value(text: str, category: FactCategory) -> str:
    lower = text.lower()
    if category == FactCategory.CONTACT:
        return extract_email(text)
    if category == FactCategory.IDENTITY:
        for prefix in ("my name is", "call me", "i am"):
            if prefix in lower:
                return lower.split(prefix, 1)[1].strip()
    if category == FactCategory.PREFERENCE:
        for prefix in ("i prefer", "prefers", "favorite", "likes", "enjoys"):
            if prefix in lower:
                return lower.split(prefix, 1)[1].strip()
    return normalize_text(text)


def first_non_empty(items: Sequence[str]) -> str:
    for item in items:
        if item:
            return item
    return ""
