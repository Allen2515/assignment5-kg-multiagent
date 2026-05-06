from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

for _k in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if _k in os.environ:
        del os.environ[_k]


@dataclass
class Intent:
    question_type: str
    keywords: list[str]
    aspect: str
    ambiguous: bool = False


_SYNONYMS: dict[str, str] = {
    "invigilator": "proctor",
    "invigilators": "proctor",
}

_STOP_WORDS = {
    "what", "is", "the", "a", "an", "for", "of", "in", "to", "how", "many",
    "can", "i", "my", "are", "be", "this", "that", "which", "will", "if",
    "before", "after", "when", "under", "at", "on", "by", "from", "with",
    "such", "as", "it", "its", "was", "not", "no", "or", "and", "but",
    "does", "do", "did", "has", "have", "had", "may", "might", "should",
    "would", "could", "than", "more", "less", "also", "only", "any",
    "them", "they", "their", "student", "students", "take", "taken",
    "get", "give", "given", "being", "using", "during", "you",
    "then", "too", "just", "about", "up", "out", "there", "want",
    "some", "like", "know", "go", "so", "now", "new", "need", "time",
    "right", "well", "good", "fine", "bit", "kind", "sort", "half",
}


class NLUnderstandingAgent:
    """Convert a natural-language question into a structured Intent."""

    def run(self, question: str) -> Intent:
        q = question.lower()

        question_type = "general"
        if re.search(r"\b(penalty|penalt|punish|consequence|happen|violation|zero|deduct|threatens?)\b", q):
            question_type = "penalty"
        elif re.search(r"\b(fee|cost|price|charge|ntd|replacing|replacement|lost)\b", q):
            question_type = "fee"
        elif re.search(r"\b(how many|minimum|maximum|at least|at most|total|credits|semesters|required|counted)\b", q):
            question_type = "quantity"
        elif re.search(r"\b(allowed|can i|may i|permitted|leave|allowed to|take.*out)\b", q):
            question_type = "eligibility"
        elif re.search(r"\b(how long|duration|period|years?|semesters?|days?|minutes?|hours?|working days?|standard)\b", q):
            question_type = "duration"
        elif re.search(r"\b(passing score|pass|failing|dismissed|expelled|poor grade|make.up)\b", q):
            question_type = "academic"

        tokens = re.findall(r"\b[\w]+\b", q)
        keywords = [
            t for t in tokens
            if t not in _STOP_WORDS and (len(t) > 2 or t.isdigit())
        ]

        ambiguous = bool(re.search(
            r"\b(maybe|probably|generally|overall|everything|entire|every|word.by.word|bit late|kind of|sort of)\b",
            q,
        ))

        # Synonym expansion
        expanded: list[str] = []
        for kw in keywords:
            expanded.append(kw)
            if kw in _SYNONYMS and _SYNONYMS[kw] not in expanded:
                expanded.append(_SYNONYMS[kw])
        # "working" + "days" → add "workday" so CONTAINS fallback finds "workdays"
        if "working" in expanded and "days" in expanded and "workday" not in expanded:
            expanded.append("workday")
        keywords = expanded[:10]

        return Intent(
            question_type=question_type,
            keywords=keywords,
            aspect=question_type,
            ambiguous=ambiguous,
        )


class SecurityAgent:
    """Reject queries that attempt to extract bulk data, modify the KG, or inject instructions."""

    _BLOCKED = [
        "delete",
        "drop",
        "merge",
        "create",
        "set ",
        "bypass",
        "ignore previous",
        "dump all",
        "export",
        "modify",
        "credentials",
        "word-by-word",
        "disable safety",
        "disable security",
    ]

    def run(self, question: str, intent: Intent) -> dict[str, str]:
        q = question.lower()
        for pattern in self._BLOCKED:
            if pattern in q:
                return {
                    "decision": "REJECT",
                    "reason": f"Unsafe pattern detected: '{pattern}'.",
                }
        return {"decision": "ALLOW", "reason": "Passed security check."}


def _sanitize_ft_query(terms: list[str]) -> str:
    """Build a Lucene-safe fulltext query string from keyword list."""
    clean = []
    for t in terms:
        t = re.sub(r"[+\-&|!(){}\[\]^\"~*?:\\/]", "", t)
        if t and len(t) > 1:
            clean.append(t)
    return " ".join(clean) if clean else "*"


class QueryPlannerAgent:
    """Build a Neo4j query execution plan from a structured intent."""

    def run(self, intent: Intent) -> dict[str, Any]:
        keywords = intent.keywords[:8]
        typed_q = _sanitize_ft_query(keywords[:5])
        broad_q = _sanitize_ft_query(keywords)
        return {
            "strategy": "typed_then_broad",
            "keywords": keywords,
            "aspect": intent.aspect,
            "typed_query": typed_q,
            "broad_query": broad_q,
        }


class QueryExecutionAgent:
    """Execute read-only queries against the A4 Knowledge Graph in Neo4j."""

    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
        try:
            self._driver = GraphDatabase.driver(uri, auth=auth)
            self._driver.verify_connectivity()
        except Exception as e:
            print(f"[QueryExecutionAgent] Neo4j unavailable: {e}")
            self._driver = None

    def run(self, plan: dict[str, Any]) -> dict[str, Any]:
        if self._driver is None:
            return {"rows": [], "error": "neo4j_unavailable"}

        keywords = plan.get("keywords", [])
        strategy = plan.get("strategy", "typed_then_broad")
        typed_q = plan.get("typed_query") or _sanitize_ft_query(keywords[:5])
        broad_q = plan.get("broad_query") or _sanitize_ft_query(keywords)

        if not keywords:
            return {"rows": [], "error": None}

        rows: list[dict] = []
        seen_ids: set[str] = set()
        last_error: str | None = None

        try:
            with self._driver.session() as session:
                if strategy in ("typed_then_broad", "fulltext_only"):
                    # Primary: rule-level fulltext index
                    try:
                        for rec in session.run(
                            'CALL db.index.fulltext.queryNodes("rule_idx", $q) '
                            "YIELD node, score "
                            "MATCH (a:Article)-[:CONTAINS_RULE]->(node) "
                            "RETURN node.rule_id AS rule_id, node.type AS type, "
                            "       node.action AS action, node.result AS result, "
                            "       node.art_ref AS art_ref, node.reg_name AS reg_name, "
                            "       a.content AS article_content, score "
                            "ORDER BY score DESC LIMIT 5",
                            q=typed_q,
                        ):
                            rid = rec.get("rule_id")
                            if rid and rid not in seen_ids:
                                seen_ids.add(rid)
                                rows.append(dict(rec))
                    except Exception as e:
                        last_error = str(e)

                    # Secondary: article-level fulltext index
                    try:
                        for rec in session.run(
                            'CALL db.index.fulltext.queryNodes("article_content_idx", $q) '
                            "YIELD node, score "
                            "MATCH (node)-[:CONTAINS_RULE]->(r:Rule) "
                            "RETURN r.rule_id AS rule_id, r.type AS type, "
                            "       r.action AS action, r.result AS result, "
                            "       r.art_ref AS art_ref, r.reg_name AS reg_name, "
                            "       node.content AS article_content, score "
                            "ORDER BY score DESC LIMIT 5",
                            q=broad_q,
                        ):
                            rid = rec.get("rule_id")
                            if rid and rid not in seen_ids:
                                seen_ids.add(rid)
                                rows.append(dict(rec))
                    except Exception as e:
                        if not last_error:
                            last_error = str(e)

                # Fallback: CONTAINS keyword scan
                if not rows:
                    for kw in keywords[:4]:
                        try:
                            for rec in session.run(
                                "MATCH (a:Article) "
                                "WHERE toLower(a.content) CONTAINS toLower($kw) "
                                "MATCH (a)-[:CONTAINS_RULE]->(r:Rule) "
                                "RETURN r.rule_id AS rule_id, r.type AS type, "
                                "       r.action AS action, r.result AS result, "
                                "       r.art_ref AS art_ref, r.reg_name AS reg_name, "
                                "       a.content AS article_content, 1.0 AS score "
                                "LIMIT 5",
                                kw=kw,
                            ):
                                rid = rec.get("rule_id")
                                if rid and rid not in seen_ids:
                                    seen_ids.add(rid)
                                    rows.append(dict(rec))
                            if rows:
                                break
                        except Exception as e:
                            if not last_error:
                                last_error = str(e)

        except Exception as e:
            return {"rows": [], "error": str(e)}

        if last_error and not rows:
            return {"rows": [], "error": last_error}

        return {"rows": rows, "error": None}


class DiagnosisAgent:
    """Classify the outcome of a query execution."""

    def run(self, execution: dict[str, Any]) -> dict[str, str]:
        error = execution.get("error")
        rows = execution.get("rows", [])

        if error and not rows:
            err_lower = str(error).lower()
            if any(k in err_lower for k in ("schema", "property", "index", "not found", "unknown")):
                return {"label": "SCHEMA_MISMATCH", "reason": str(error)}
            return {"label": "QUERY_ERROR", "reason": str(error)}

        if not rows:
            return {"label": "NO_DATA", "reason": "No matching rule found in KG."}

        # Validate that returned rows carry the expected A4 schema fields
        required = {"rule_id", "action", "result"}
        for row in rows[:2]:
            present = {k for k, v in row.items() if v is not None}
            missing = required - present
            if missing:
                return {"label": "SCHEMA_MISMATCH", "reason": f"Missing row fields: {missing}"}

        return {"label": "SUCCESS", "reason": "Query returned valid rule evidence."}


class QueryRepairAgent:
    """Produce a revised query plan when the original plan fails."""

    def run(
        self,
        diagnosis: dict[str, str],
        original_plan: dict[str, Any],
        intent: Intent,
    ) -> dict[str, Any]:
        repaired = dict(original_plan)
        repaired["strategy"] = "fulltext_only"

        # Use only longer, less ambiguous keywords to reduce Lucene parse errors
        simplified = [k for k in intent.keywords if len(k) >= 4][:5]
        if not simplified:
            simplified = intent.keywords[:3]

        repaired["keywords"] = simplified
        repaired["typed_query"] = _sanitize_ft_query(simplified[:3])
        repaired["broad_query"] = _sanitize_ft_query(simplified)
        return repaired


class ExplanationAgent:
    """Generate a human-readable explanation of the full pipeline execution."""

    def run(
        self,
        question: str,
        intent: Intent,
        security: dict[str, str],
        diagnosis: dict[str, str],
        answer: str,
        repair_attempted: bool,
    ) -> str:
        return (
            f"[NLU] type={intent.question_type}, keywords={intent.keywords[:5]}. "
            f"[Security] {security['decision']}: {security['reason']} "
            f"[Diagnosis] {diagnosis['label']}: {diagnosis['reason']} "
            f"[Repair] attempted={repair_attempted}. "
            f"[Answer] {answer[:100]}"
        )


def build_template_pipeline() -> dict[str, Any]:
    """Factory that instantiates all agents and returns the pipeline dict."""
    return {
        "nlu": NLUnderstandingAgent(),
        "security": SecurityAgent(),
        "planner": QueryPlannerAgent(),
        "executor": QueryExecutionAgent(),
        "diagnosis": DiagnosisAgent(),
        "repair": QueryRepairAgent(),
        "explanation": ExplanationAgent(),
    }
