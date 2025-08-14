"""
utils.py

Small utilities for:
- Robustly parsing model outputs (lists, code blocks, SQL snippets),
- Extracting fenced code,
- Fuzzy-matching categorical filter values against actual DB values.

Important:
- Functions here are side-effect free except fuzzy matchers, which read the DB.
"""

from __future__ import annotations
import ast
import json
import re
from typing import List

import pandas as pd
from sqlalchemy import text

from config import get_engine

# -------------- Parsing helpers --------------
def parse_nested_list(text_in: str) -> list:
    """Parse model output into a Python list; tries JSON, then literal_eval, then bracket extraction."""
    if not text_in:
        return []
    s = str(text_in).strip()
    # Try JSON
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # Try Python literal
    try:
        obj = ast.literal_eval(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        pass
    # Fallback: first top-level [ [ ... ], ... ] pattern
    m = re.search(r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]", s, re.DOTALL)
    if m:
        try:
            obj = ast.literal_eval(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []
    return []

def normalize_subquestions(entries: list) -> List[List[str]]:
    """Ensure each entry is exactly [subquestion, table]."""
    norm: List[List[str]] = []
    for e in entries:
        if isinstance(e, list) and len(e) >= 2:
            subq = str(e[0]).strip()
            table = str(e[1]).strip()
            if subq and table:
                norm.append([subq, table])
    return norm

def extract_sql(text_in: str) -> str:
    """
    Extract SQL from a ```sql ...``` fenced block, else from first SELECT onward, else raw stripped.
    """
    if not text_in:
        return ""
    s = str(text_in)
    # Prefer fenced ```sql ... ```
    m = re.search(r"```(?:\s*sql)?\s*(.*?)```", s, flags=re.I | re.S)
    if m:
        return m.group(1).strip()
    # Else from first SELECT
    m = re.search(r"(?is)\bselect\b.*", s)
    if m:
        return m.group(0).strip()
    return s.strip()

# -------------- Code block extraction --------------
def extract_code_block(content: str, language: str) -> str:
    """
    Extract code from a fenced block: ```<language> ... ```
    If not found, return the first fenced block or content without backticks.
    """
    if content is None:
        return ""
    s = str(content)
    # ```language ... ```
    pattern_lang = rf"```(?:\s*{re.escape(language)})\s*(.*?)```"
    m = re.search(pattern_lang, s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # First fenced block
    m = re.search(r"```(.*?)```", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    return s.replace("```", "").strip()

# -------------- Fuzzy filter matcher --------------
_engine = get_engine()

def _get_values(table_name: str, column_name: str):
    q = text(f"SELECT DISTINCT {column_name} AS v FROM {table_name}")
    df = pd.read_sql(q, con=_engine)
    return df["v"].dropna().astype(str).tolist()

def _best_fuzzy_match(input_value: str, choices):
    #  replacement for rapidfuzz token_set_ratio using simple heuristic if RF not installed
    try:
        from rapidfuzz import process, fuzz
        match, score, _ = process.extractOne(input_value, choices, scorer=fuzz.token_set_ratio)
        return match, score
    except Exception:
        # Very light fallback: exact or casefold match, else return original
        s = str(input_value).casefold()
        for c in choices:
            if s == str(c).casefold():
                return c, 100
        return input_value, 0

def _flatten_filters_structure(filters):
    """
    Accept either:
      ["yes", ["t","c","v"], ["t","c","v"], ...]
    or the nested variant:
      ["yes", [ ["t","c","v"], ["t","c","v"], ... ]]
    and normalize to the flat version.
    """
    if (
        isinstance(filters, list)
        and filters
        and filters[0] == "yes"
        and len(filters) == 2
        and isinstance(filters[1], list)
        and filters[1]
        and all(isinstance(x, list) for x in filters[1])
    ):
        return ["yes", *filters[1]]
    return filters

def fuzzy_match_filters(filters):
    """
    For categorical equality-like predicates (no operators), fuzzy-match the value
    to the column's distinct set. Otherwise pass through unchanged.

    Input:
      ["yes", ["table","column","predicate"], ...]  or
      ["yes", [ ["table","column","predicate"], ... ]]
    Output (same shape, normalized to flat):
      ["yes", ["table","column","matched_value"], ...]
    """
    if not isinstance(filters, list) or not filters or filters[0] == "no":
        return filters
    filters = _flatten_filters_structure(filters)
    out = ["yes"]
    for t in filters[1:]:
        if not isinstance(t, list) or len(t) < 3:
            continue
        table, column, predicate = t[0], t[1], str(t[2]).strip()
        # textual equality-like predicate (no ranges/dates/operators)
        if re.search(r"[A-Za-z]", predicate) and not re.search(
            r"\bbetween\b|<=|>=|<|>|before|after|\d{4}-\d{2}-\d{2}", predicate, re.I
        ):
            choices = _get_values(table, column)
            best, _ = _best_fuzzy_match(predicate, choices) if choices else (predicate, 0)
            out.append([table, column, best])
        else:
            out.append([table, column, predicate])
    return out
