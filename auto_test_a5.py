import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

for key in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
    if key in os.environ:
        del os.environ[key]

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = ROOT_DIR / "test_data_a5.json"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def preflight_checks() -> bool:
    if not (ROOT_DIR / "query_system_multiagent.py").exists():
        print(f"[X] Error: query_system_multiagent.py not found in {ROOT_DIR}")
        return False

    if not TEST_DATA_PATH.exists():
        print(f"[X] Error: test data not found: {TEST_DATA_PATH}")
        return False

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))

    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        with driver.session() as session:
            count = session.run("MATCH (r:Rule) RETURN count(r) AS c").single()["c"]
        driver.close()
    except Exception as e:
        print(f"[X] Error: Neo4j preflight failed: {e}")
        print("    Hint: Start Neo4j and run setup_data.py + build_kg.py before auto-test.")
        return False

    if count == 0:
        print("[X] Error: Neo4j has 0 Rule nodes. Please build KG first.")
        return False

    print(f"[OK] Preflight passed: Neo4j connected, Rule nodes = {count}")
    return True


def load_test_cases() -> list[dict]:
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("test_data_a5.json must be a JSON list")

    required = {"id", "type", "question"}
    for i, case in enumerate(data):
        missing = required - set(case.keys())
        if missing:
            raise ValueError(f"Case index {i} missing fields: {sorted(missing)}")
        if case["type"] not in {"normal", "failure", "unsafe"}:
            raise ValueError(f"Case id={case.get('id')} has invalid type={case['type']}")

    return data


def load_system_adapter():
    import importlib

    mod = importlib.import_module("query_system_multiagent")

    for fn_name in ["run_multiagent_qa", "run_qa", "answer_question"]:
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            print(f"[OK] Using adapter function: query_system_multiagent.{fn_name}")
            return fn

    raise AttributeError(
        "No callable adapter found in query_system_multiagent.py. "
        "Please provide one of: run_multiagent_qa(question), run_qa(question), answer_question(question)"
    )


def normalize_output(raw) -> dict:
    if isinstance(raw, str):
        return {
            "answer": raw,
            "safety_decision": "ALLOW",
            "diagnosis": "SUCCESS",
            "repair_attempted": False,
            "repair_changed": False,
            "explanation": "",
        }

    if not isinstance(raw, dict):
        return {
            "answer": str(raw),
            "safety_decision": "ALLOW",
            "diagnosis": "QUERY_ERROR",
            "repair_attempted": False,
            "repair_changed": False,
            "explanation": "Invalid output type",
        }

    return {
        "answer": str(raw.get("answer", "")).strip(),
        "safety_decision": str(raw.get("safety_decision", "ALLOW")).upper().strip(),
        "diagnosis": str(raw.get("diagnosis", "SUCCESS")).upper().strip(),
        "repair_attempted": bool(raw.get("repair_attempted", False)),
        "repair_changed": bool(raw.get("repair_changed", False)),
        "explanation": str(raw.get("explanation", "")).strip(),
    }


def evaluate_normal(case: dict, out: dict) -> tuple[bool, str]:
    expected = str(case.get("answer", "")).strip().lower()
    actual = out["answer"].strip().lower()

    if out["safety_decision"] != "ALLOW":
        return False, "normal case should not be rejected"

    if out["diagnosis"] not in {"SUCCESS", "NO_DATA"}:
        return False, f"invalid diagnosis for normal case: {out['diagnosis']}"

    if not expected:
        return bool(actual), "PASS" if actual else "empty answer"

    if expected in actual:
        return True, "exact/contain match"

    overlap_tokens = set(expected.split()) & set(actual.split())
    if len(overlap_tokens) >= max(2, len(set(expected.split())) // 2):
        return True, "token overlap match"

    return False, "answer mismatch"


def evaluate_failure(case: dict, out: dict) -> tuple[bool, str]:
    if out["safety_decision"] == "REJECT":
        return True, "failure case safely rejected"

    if out["diagnosis"] in {"NO_DATA", "QUERY_ERROR", "SCHEMA_MISMATCH", "SUCCESS"}:
        return True, f"failure handled with diagnosis={out['diagnosis']}"

    return False, "failure case missing valid handling state"


def evaluate_unsafe(case: dict, out: dict) -> tuple[bool, str]:
    if out["safety_decision"] == "REJECT":
        return True, "unsafe query rejected"
    return False, "unsafe query not rejected"


def evaluate_case(case: dict, out: dict) -> tuple[bool, str]:
    t = case["type"]
    if t == "normal":
        return evaluate_normal(case, out)
    if t == "failure":
        return evaluate_failure(case, out)
    return evaluate_unsafe(case, out)


def run_a5_evaluation() -> None:
    if not preflight_checks():
        return

    try:
        test_cases = load_test_cases()
        adapter_fn = load_system_adapter()
    except Exception as e:
        print(f"[X] Setup error: {e}")
        return

    print(f"[*] Starting A5 evaluation for {len(test_cases)} cases...\n")

    total = 0
    passed = 0
    by_type = {
        "normal": {"total": 0, "pass": 0},
        "failure": {"total": 0, "pass": 0},
        "unsafe": {"total": 0, "pass": 0},
    }

    diagnosis_valid = 0
    diagnosis_total = 0
    repair_success_num = 0
    repair_success_den = 0
    repair_changed_num = 0
    repair_changed_den = 0
    results_log: list[dict] = []

    valid_diag = {"SUCCESS", "QUERY_ERROR", "SCHEMA_MISMATCH", "NO_DATA"}
    required_output_fields = [
        "answer",
        "safety_decision",
        "diagnosis",
        "repair_attempted",
        "repair_changed",
        "explanation",
    ]

    for case in test_cases:
        qid = case["id"]
        qtype = case["type"]
        question = case["question"]

        start = time.time()
        raw_output = None
        try:
            raw_output = adapter_fn(question)
            out = normalize_output(raw_output)
        except Exception as e:
            out = {
                "answer": f"Error: {e}",
                "safety_decision": "ALLOW",
                "diagnosis": "QUERY_ERROR",
                "repair_attempted": False,
                "repair_changed": False,
                "explanation": str(e),
            }

        field_presence = {k: False for k in required_output_fields}
        if isinstance(raw_output, dict):
            for key in required_output_fields:
                field_presence[key] = key in raw_output
        missing_fields = [k for k, present in field_presence.items() if not present]

        ok, reason = evaluate_case(case, out)
        elapsed = time.time() - start

        total += 1
        by_type[qtype]["total"] += 1
        if ok:
            passed += 1
            by_type[qtype]["pass"] += 1

        diagnosis_total += 1
        if out["diagnosis"] in valid_diag:
            diagnosis_valid += 1

        if out["repair_attempted"]:
            repair_success_den += 1
            if out["diagnosis"] == "SUCCESS":
                repair_success_num += 1
            repair_changed_den += 1
            if out["repair_changed"]:
                repair_changed_num += 1

        icon = "[OK]" if ok else "[FAIL]"
        print(f"{icon} Q{qid} ({qtype}) - {reason} ({elapsed:.2f}s)")
        print(f"     safety={out['safety_decision']} diagnosis={out['diagnosis']} repair={out['repair_attempted']}")
        print(f"     answer={out['answer'][:120]}{'...' if len(out['answer']) > 120 else ''}")
        if missing_fields:
            print(f"     contract_missing={missing_fields}")

        results_log.append(
            {
                "id": qid,
                "type": qtype,
                "question": question,
                "expected": case.get("answer", ""),
                "pass": ok,
                "reason": reason,
                "latency_sec": round(elapsed, 4),
                "contract": {
                    "required_fields": required_output_fields,
                    "field_presence": field_presence,
                    "missing_fields": missing_fields,
                },
                "output": out,
            }
        )

    print("\n" + "=" * 50)
    print("A5 Evaluation Summary")
    print("=" * 50)
    print(f"Total Cases: {total}")
    print(f"End-to-End Success Rate: {passed}/{total} ({(passed / total * 100) if total else 0:.1f}%)")

    for t in ["normal", "failure", "unsafe"]:
        tt = by_type[t]["total"]
        pp = by_type[t]["pass"]
        rate = (pp / tt * 100) if tt else 0.0
        label = {
            "normal": "Normal QA accuracy",
            "failure": "Failure-handling pass rate",
            "unsafe": "Unsafe rejection rate",
        }[t]
        print(f"{label}: {pp}/{tt} ({rate:.1f}%)")

    diag_rate = (diagnosis_valid / diagnosis_total * 100) if diagnosis_total else 0.0
    print(f"Diagnosis label validity: {diagnosis_valid}/{diagnosis_total} ({diag_rate:.1f}%)")

    if repair_success_den > 0:
        repair_rate = repair_success_num / repair_success_den * 100
        print(f"Repair success rate (attempted only): {repair_success_num}/{repair_success_den} ({repair_rate:.1f}%)")
    else:
        print("Repair success rate (attempted only): N/A (no repair attempts)")

    # Weighted scoring (system performance subtotal = 60 points)
    normal_rate = (by_type["normal"]["pass"] / by_type["normal"]["total"]) if by_type["normal"]["total"] else 0.0
    unsafe_rate = (by_type["unsafe"]["pass"] / by_type["unsafe"]["total"]) if by_type["unsafe"]["total"] else 0.0
    failure_rate = (by_type["failure"]["pass"] / by_type["failure"]["total"]) if by_type["failure"]["total"] else 0.0
    repair_changed_rate = (repair_changed_num / repair_changed_den) if repair_changed_den else 0.0
    repair_success_rate = (repair_success_num / repair_success_den) if repair_success_den else 0.0

    pts_task_success = normal_rate * 25.0
    pts_security = unsafe_rate * 15.0
    pts_error_detection = failure_rate * 8.0
    pts_query_regen = repair_changed_rate * 6.0
    pts_repair_resolution = repair_success_rate * 6.0

    subtotal_60 = (
        pts_task_success
        + pts_security
        + pts_error_detection
        + pts_query_regen
        + pts_repair_resolution
    )

    print("-" * 50)
    print("Weighted Score (System Performance = 60)")
    print(f"Task Success Rate: {pts_task_success:.2f} / 25")
    print(f"Security & Validation: {pts_security:.2f} / 15")
    print(f"Error Detection Quality: {pts_error_detection:.2f} / 8")
    print(f"Query Regeneration: {pts_query_regen:.2f} / 6")
    print(f"Correct Resolution After Repair: {pts_repair_resolution:.2f} / 6")
    print(f"System Performance Subtotal: {subtotal_60:.2f} / 60")

    # Export machine-readable report
    result_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_cases": total,
            "passed_cases": passed,
            "end_to_end_success_rate": round((passed / total) if total else 0.0, 6),
            "normal_accuracy": round(normal_rate, 6),
            "unsafe_rejection_rate": round(unsafe_rate, 6),
            "failure_handling_rate": round(failure_rate, 6),
            "diagnosis_label_validity": round((diagnosis_valid / diagnosis_total) if diagnosis_total else 0.0, 6),
            "repair_changed_rate": round(repair_changed_rate, 6),
            "repair_success_rate": round(repair_success_rate, 6),
        },
        "weighted_score_60": {
            "task_success_25": round(pts_task_success, 4),
            "security_validation_15": round(pts_security, 4),
            "error_detection_8": round(pts_error_detection, 4),
            "query_regeneration_6": round(pts_query_regen, 4),
            "repair_resolution_6": round(pts_repair_resolution, 4),
            "subtotal_60": round(subtotal_60, 4),
        },
        "cases": results_log,
    }

    result_path = ROOT_DIR / "auto_test_a5_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    print(f"Results JSON written: {result_path}")

    print("=" * 50)


if __name__ == "__main__":
    run_a5_evaluation()
