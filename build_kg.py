"""KG builder for Assignment 5 (A4 carry-over).

Graph schema (unchanged from A4):
  (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)

Article props : number, content, reg_name, category
Rule props    : rule_id, type, action, result, art_ref, reg_name
Fulltext idx  : article_content_idx, rule_idx
SQLite source : ncu_regulations.db  (created by setup_data.py)
"""

import json
import os
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)


def _parse_llm_json(text: str) -> list[dict]:
    """Try multiple strategies to extract a rules list from LLM output."""
    try:
        data = json.loads(text)
        if isinstance(data.get("rules"), list):
            return data["rules"]
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data.get("rules"), list):
                return data["rules"]
        except Exception:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            items = json.loads(text[start : end + 1])
            if isinstance(items, list):
                return items
        except Exception:
            pass

    return []


def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    """Deterministic sentence-based rule extraction used when LLM fails."""
    if not content:
        return []

    rules: list[dict[str, str]] = []
    sentences = re.split(r"(?<=[.;:])\s+", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]

    for sent in sentences[:10]:
        rule_type = "general"
        if re.search(r"(penalty|penalt|fine|deduct|forfeit|zero|disciplin|expel|dismiss)", sent, re.I):
            rule_type = "penalty"
        elif re.search(r"(fee|NTD|NT\$|cost|charge|replac)", sent, re.I):
            rule_type = "fee"
        elif re.search(r"(required|must|shall|minimum|at least|mandatory)", sent, re.I):
            rule_type = "requirement"
        elif re.search(r"(year|semester|month|week|working day|hour|minute)", sent, re.I):
            rule_type = "duration"
        elif re.search(r"(credit|grade|score|point|GPA)", sent, re.I):
            rule_type = "academic"
        elif re.search(r"(allowed|permitted|may|can)", sent, re.I):
            rule_type = "eligibility"

        parts = re.split(r"(?:,?\s*(?:will|shall)\s+)", sent, maxsplit=1)
        if len(parts) == 2 and len(parts[0]) >= 10 and len(parts[1]) >= 10:
            action, result = parts[0].strip(), parts[1].strip()
        else:
            action = result = sent

        rules.append({
            "type": rule_type,
            "action": action[:300],
            "result": result[:300],
        })

    rules.insert(0, {
        "type": "general",
        "action": f"{article_number} {content[:250]}",
        "result": content[:250],
    })

    return rules[:7]


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """Use local LLM to extract structured rules; fall back to deterministic extraction."""
    pipe = get_raw_pipeline()
    tok = get_tokenizer()

    if pipe is None or tok is None:
        return {"rules": build_fallback_rules(article_number, content)}

    system_msg = (
        "You are a rule extractor for university regulations. "
        "Read the article text and output ONLY a JSON object. "
        'Format: {"rules": [{"type": "...", "action": "...", "result": "..."}]}\n'
        "type: one of penalty | requirement | fee | duration | condition | procedure\n"
        "action: the condition, trigger, or behavior described in the rule\n"
        "result: the consequence, outcome, or requirement stated\n"
        "Output ONLY the JSON object with no additional text."
    )

    user_msg = (
        f"Article: {article_number}\n"
        f"Regulation: {reg_name}\n"
        f"Content: {content[:900]}\n\n"
        "Extract all rules. Output ONLY JSON:"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = pipe(prompt, max_new_tokens=512)[0]["generated_text"].strip()
        parsed = _parse_llm_json(raw)
        if parsed:
            return {"rules": parsed}
    except Exception as e:
        print(f"  [LLM extraction warning – {article_number}]: {e}")

    return {"rules": build_fallback_rules(article_number, content)}


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    load_local_llm()

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0
        total_articles = len(articles)

        for idx, (reg_id, article_number, content) in enumerate(articles, 1):
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            print(f"  [{idx}/{total_articles}] Extracting rules: {reg_name} – {article_number}")

            entity_data = extract_entities(article_number, reg_name, content)
            rules = entity_data.get("rules", [])

            if not rules:
                rules = build_fallback_rules(article_number, content)

            seen_keys: set[str] = set()
            unique_rules: list[dict] = []
            for rule in rules:
                key = re.sub(r"\s+", " ", rule.get("action", ""))[:80].lower().strip()
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    unique_rules.append(rule)

            for rule in unique_rules:
                action = rule.get("action", "").strip()
                result = rule.get("result", "").strip()

                if not action or not result or len(action) < 5:
                    continue

                rule_counter += 1
                safe_art = re.sub(r"[^A-Za-z0-9_]", "_", article_number)
                rule_id = f"R{reg_id}_{safe_art}_{rule_counter}"

                session.run(
                    """
                    MATCH (a:Article {number: $art_num, reg_name: $rname})
                    CREATE (r:Rule {
                        rule_id: $rule_id,
                        type:    $rtype,
                        action:  $action,
                        result:  $result,
                        art_ref: $art_ref,
                        reg_name: $rname
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    art_num=article_number,
                    rname=reg_name,
                    rule_id=rule_id,
                    rtype=rule.get("type", "general"),
                    action=action[:500],
                    result=result[:500],
                    art_ref=rule.get("art_ref", article_number),
                )

        print(f"\n[Rule creation] Total rules created: {rule_counter}")

        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_a = int((coverage or {}).get("total_articles", 0) or 0)
        covered_a = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_a = int((coverage or {}).get("uncovered_articles", 0) or 0)
        print(f"[Coverage] covered={covered_a}/{total_a}, uncovered={uncovered_a}")

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()
