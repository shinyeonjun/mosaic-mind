"""One-step counterfactual analysis for reward-only active probing.

For a surviving hypothesis set S and probe q, estimate
    E[correct after q] - E[correct now] - probe_cost
under both a uniform prior over S and the held-out fold's rule prior. The
final-case-specific probe search is reported separately as an oracle ceiling.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random

from cognitive_lab.active.baselines import choose_answer, select_probe, update_posterior
from cognitive_lab.active.controller import FOLD_TEST_RULES
from cognitive_lab.active.environment import (
    ALL_PROBES,
    PROBE_OUTCOMES,
    RULES,
    TaskCase,
    apply_case,
    generate_case,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"
DEFAULT_INPUT = RESULTS_DIR / "experiment-006B_failures_all.jsonl"
CASE_DISTRIBUTION_SEED = 60042
CONFIDENCE_LEVEL = 0.95


def _probe_json(probe_id: int) -> list[list[int | bool]]:
    return [[is_target, key] for is_target, key in ALL_PROBES[probe_id]]


def _posterior_mask(rule_ids: tuple[int, ...]) -> int:
    return sum(1 << rule_id for rule_id in rule_ids)


def _expected_accuracy_by_posterior(
    cases: list[TaskCase],
) -> dict[int, dict[int, float]]:
    """Estimate answer accuracy for every posterior, conditioned on hidden rule."""
    answer_rows = [
        tuple(apply_case(rule, case) for rule in RULES)
        for case in cases
    ]
    accuracy: dict[int, dict[int, float]] = {}
    for mask in range(1, 1 << len(RULES)):
        members = tuple(rule_id for rule_id in range(len(RULES)) if mask & (1 << rule_id))
        correct_by_hidden_rule = Counter()
        for answers in answer_rows:
            counts = Counter(answers[rule_id] for rule_id in members)
            highest = max(counts.values())
            prediction = min(answer for answer, count in counts.items() if count == highest)
            for rule_id in members:
                correct_by_hidden_rule[rule_id] += int(answers[rule_id] == prediction)
        accuracy[mask] = {
            rule_id: correct_by_hidden_rule[rule_id] / len(cases)
            for rule_id in members
        }
    return accuracy


def _expected_net_gain_by_probe(
    accuracy: dict[int, dict[int, float]], probe_cost: float
) -> tuple[dict[int, dict[int, float]], dict[tuple[int, int], dict[int, float]]]:
    """Expected gain under each prior, keeping predictions on the full candidate set."""
    gains: dict[int, dict[int, float]] = {}
    fold_gains: dict[tuple[int, int], dict[int, float]] = {}
    for mask, correct_rates in accuracy.items():
        members = tuple(correct_rates)
        current_accuracy = sum(correct_rates[rule_id] for rule_id in members) / len(members)
        fold_members = {
            fold: tuple(rule_id for rule_id in members if rule_id in test_rules)
            for fold, test_rules in enumerate(FOLD_TEST_RULES)
        }
        fold_current_accuracy = {
            fold: sum(correct_rates[rule_id] for rule_id in supported) / len(supported)
            for fold, supported in fold_members.items()
            if supported
        }
        by_probe = {}
        by_fold: dict[int, dict[int, float]] = {
            fold: {} for fold in fold_current_accuracy
        }
        for probe_id in range(len(ALL_PROBES)):
            posterior_by_outcome: dict[int, tuple[int, ...]] = {}
            for rule_id in members:
                outcome = PROBE_OUTCOMES[probe_id][rule_id]
                if outcome not in posterior_by_outcome:
                    posterior_by_outcome[outcome] = tuple(
                        candidate for candidate in members
                        if PROBE_OUTCOMES[probe_id][candidate] == outcome
                    )
            expected_accuracy = sum(
                accuracy[
                    _posterior_mask(
                        posterior_by_outcome[PROBE_OUTCOMES[probe_id][rule_id]]
                    )
                ][rule_id]
                for rule_id in members
            ) / len(members)
            by_probe[probe_id] = expected_accuracy - current_accuracy - probe_cost
            for fold, supported in fold_members.items():
                if not supported:
                    continue
                fold_expected_accuracy = sum(
                    accuracy[
                        _posterior_mask(
                            posterior_by_outcome[PROBE_OUTCOMES[probe_id][rule_id]]
                        )
                    ][rule_id]
                    for rule_id in supported
                ) / len(supported)
                by_fold[fold][probe_id] = (
                    fold_expected_accuracy - fold_current_accuracy[fold] - probe_cost
                )
            if len(members) == 1 and not math.isclose(
                by_probe[probe_id], -probe_cost, abs_tol=1e-12
            ):
                raise AssertionError("a query cannot improve accuracy with one candidate rule")
        gains[mask] = by_probe
        fold_gains.update({(mask, fold): values for fold, values in by_fold.items()})
    return gains, fold_gains


def _source_result_for_name(source_name: str, cache: dict[str, dict]) -> dict:
    if source_name not in cache:
        if Path(source_name).name != source_name:
            raise ValueError(f"invalid source result name: {source_name!r}")
        source_path = RESULTS_DIR / source_name
        cache[source_name] = json.loads(source_path.read_text(encoding="utf-8"))
    return cache[source_name]


def _case_oracle_counterfactual(
    case: TaskCase,
    expected: int,
    posterior: tuple[int, ...],
    hidden_rule_id: int,
    asked: set[int],
) -> tuple[list[int], int | None]:
    """Find probes that fix this already-known case; diagnostic upper bound only."""
    fixing = []
    for probe_id in range(len(ALL_PROBES)):
        if probe_id in asked:
            continue
        outcome = PROBE_OUTCOMES[probe_id][hidden_rule_id]
        updated = update_posterior(posterior, probe_id, outcome)
        if choose_answer(case, updated) == expected:
            fixing.append(probe_id)
    return fixing, fixing[0] if fixing else None


def _analyze_failure(
    record: dict,
    accuracy: dict[int, dict[int, float]],
    expected_gains: dict[int, dict[int, float]],
    fold_expected_gains: dict[tuple[int, int], dict[int, float]],
    probe_cost: float,
    confidence_half_width: float,
) -> dict:
    hidden_rule_id = record["hidden_rule_id"]
    if type(hidden_rule_id) is not int or hidden_rule_id not in range(len(RULES)):
        raise ValueError("failure record contains an invalid hidden rule")
    case_data = record["case"]
    case = TaskCase(
        target_door=case_data["target_door"],
        events=tuple(tuple(event) for event in case_data["events"]),
    )
    expected = apply_case(RULES[hidden_rule_id], case)
    if expected != record["expected"]:
        raise ValueError("record's expected answer disagrees with its hidden rule and case")

    posterior = tuple(record["final_candidate_rule_ids"])
    if (
        not posterior
        or len(set(posterior)) != len(posterior)
        or any(type(rule_id) is not int or rule_id not in range(len(RULES)) for rule_id in posterior)
    ):
        raise ValueError("failure record contains an invalid final posterior")
    if hidden_rule_id not in posterior:
        raise ValueError("failure record eliminated its own hidden rule")
    fold = record["fold"]
    if type(fold) is not int or fold not in range(len(FOLD_TEST_RULES)):
        raise ValueError("failure record contains an invalid fold")
    fold_rules = FOLD_TEST_RULES[fold]
    fold_posterior = tuple(rule_id for rule_id in posterior if rule_id in fold_rules)
    if not fold_posterior:
        raise ValueError("failure posterior has no rule from its test fold")
    asked = {
        step["probe_id"]
        for step in record["trajectory"]
        if step.get("action") == "probe"
    }
    if any(probe_id not in range(len(ALL_PROBES)) for probe_id in asked):
        raise ValueError("failure record contains an invalid asked probe")
    available = [probe_id for probe_id in range(len(ALL_PROBES)) if probe_id not in asked]
    if not available:
        raise ValueError("no unasked probes remain")

    mask = _posterior_mask(posterior)
    gains = expected_gains[mask]
    best_probe = min(available, key=lambda probe_id: (-gains[probe_id], probe_id))
    best_gain = gains[best_probe]
    if best_gain - confidence_half_width > 0:
        one_step_decision = "query_supported"
    elif best_gain + confidence_half_width < 0:
        one_step_decision = "stop_supported"
    else:
        one_step_decision = "uncertain"

    fold_gains = fold_expected_gains[(mask, fold)]
    fold_best_probe = min(
        available, key=lambda probe_id: (-fold_gains[probe_id], probe_id)
    )
    fold_best_gain = fold_gains[fold_best_probe]
    if fold_best_gain - confidence_half_width > 0:
        fold_best_decision = "query_supported"
    elif fold_best_gain + confidence_half_width < 0:
        fold_best_decision = "stop_supported"
    else:
        fold_best_decision = "uncertain"

    information_gain_probe = select_probe(
        "information_gain", posterior, asked, random.Random(0)
    )
    if information_gain_probe is None:
        information_gain_after = posterior
        information_gain_prediction = choose_answer(case, posterior)
        information_gain_expected_gain = None
        fold_information_gain_expected_gain = None
    else:
        outcome = PROBE_OUTCOMES[information_gain_probe][hidden_rule_id]
        information_gain_after = update_posterior(posterior, information_gain_probe, outcome)
        information_gain_prediction = choose_answer(case, information_gain_after)
        information_gain_expected_gain = gains[information_gain_probe]
        fold_information_gain_expected_gain = fold_gains[information_gain_probe]

    fixing_probe_ids, first_fixing_probe = _case_oracle_counterfactual(
        case, expected, posterior, hidden_rule_id, asked
    )
    if information_gain_prediction == expected:
        case_outcome = "information_gain_probe_fixes_case"
    elif fixing_probe_ids:
        case_outcome = "only_case_oracle_finds_a_fix"
    else:
        case_outcome = "no_single_probe_fixes_case"

    return {
        "posterior_size": len(posterior),
        "uniform_candidate_reference": {
            "current_stop_accuracy_estimate": round(
                sum(accuracy[mask][rule_id] for rule_id in posterior) / len(posterior),
                6,
            ),
            "expected_gain_of_best_case_independent_probe": round(best_gain, 6),
            "best_case_independent_probe_id": best_probe,
            "best_case_independent_probe": _probe_json(best_probe),
            "best_probe_gain_simultaneous_95pct_bound": [
                round(best_gain - confidence_half_width, 6),
                round(best_gain + confidence_half_width, 6),
            ],
            "information_gain_probe_expected_net_gain": (
                round(information_gain_expected_gain, 6)
                if information_gain_expected_gain is not None else None
            ),
            "query_stop_decision": one_step_decision,
        },
        "test_fold_conditioned_reference": {
            "fold": fold,
            "test_rule_ids": list(fold_rules),
            "surviving_test_rule_ids": list(fold_posterior),
            "current_stop_accuracy_estimate": round(
                sum(accuracy[mask][rule_id] for rule_id in fold_posterior)
                / len(fold_posterior),
                6,
            ),
            "information_gain_probe_expected_net_gain": (
                round(fold_information_gain_expected_gain, 6)
                if fold_information_gain_expected_gain is not None else None
            ),
            "information_gain_probe_gain_simultaneous_95pct_bound": (
                [
                    round(fold_information_gain_expected_gain - confidence_half_width, 6),
                    round(fold_information_gain_expected_gain + confidence_half_width, 6),
                ]
                if fold_information_gain_expected_gain is not None else None
            ),
            "fold_prior_oracle_best_probe_expected_net_gain": round(fold_best_gain, 6),
            "fold_prior_oracle_best_probe_id": fold_best_probe,
            "fold_prior_oracle_best_probe": _probe_json(fold_best_probe),
            "fold_prior_oracle_best_probe_gain_simultaneous_95pct_bound": [
                round(fold_best_gain - confidence_half_width, 6),
                round(fold_best_gain + confidence_half_width, 6),
            ],
            "fold_prior_oracle_best_query_stop_decision": fold_best_decision,
        },
        "information_gain_probe_id": information_gain_probe,
        "information_gain_probe_expected_net_gain": (
            round(information_gain_expected_gain, 6)
            if information_gain_expected_gain is not None else None
        ),
        "information_gain_probe_fixes_this_case": information_gain_prediction == expected,
        "information_gain_posterior_after_probe": list(information_gain_after),
        "case_oracle_can_find_a_fix": bool(fixing_probe_ids),
        "case_oracle_fixing_probe_count": len(fixing_probe_ids),
        "case_oracle_first_fixing_probe_id": first_fixing_probe,
        "case_oracle_first_fixing_probe": (
            _probe_json(first_fixing_probe) if first_fixing_probe is not None else None
        ),
        "case_oracle_outcome": case_outcome,
        "probe_cost": probe_cost,
    }


def _summarize(groups: dict[tuple[str, int], Counter]) -> dict:
    summary = defaultdict(dict)
    for (controller, budget), counts in sorted(groups.items()):
        episodes = counts["episodes"]
        summary[controller][str(budget)] = {
            "failure_records": episodes,
            "information_gain_probe_fixes_case": counts["information_gain_probe_fixes_case"],
            "only_case_oracle_finds_a_fix": counts["only_case_oracle_finds_a_fix"],
            "no_single_probe_fixes_case": counts["no_single_probe_fixes_case"],
            "case_oracle_fix_rate": round(counts["case_oracle_fixable"] / episodes, 4),
            "uniform_candidate_reference": {
                "best_probe_query_supported": counts["uniform_query_supported"],
                "best_probe_stop_supported": counts["uniform_stop_supported"],
                "best_probe_uncertain": counts["uniform_uncertain"],
                "mean_best_probe_expected_net_gain": round(
                    counts["uniform_best_gain"] / episodes, 6
                ),
                "mean_information_gain_probe_expected_net_gain": round(
                    counts["uniform_information_gain"] / counts["information_gain_count"], 6
                ) if counts["information_gain_count"] else None,
            },
            "test_fold_conditioned_reference": {
                "fold_prior_oracle_best_probe_query_supported": counts["fold_query_supported"],
                "fold_prior_oracle_best_probe_stop_supported": counts["fold_stop_supported"],
                "fold_prior_oracle_best_probe_uncertain": counts["fold_uncertain"],
                "mean_fold_prior_oracle_best_probe_expected_net_gain": round(
                    counts["fold_best_gain"] / episodes, 6
                ),
                "information_gain_probe_query_supported": counts["fold_ig_query_supported"],
                "information_gain_probe_stop_supported": counts["fold_ig_stop_supported"],
                "information_gain_probe_uncertain": counts["fold_ig_uncertain"],
                "mean_information_gain_probe_expected_net_gain": round(
                    counts["fold_information_gain"] / counts["information_gain_count"], 6
                ) if counts["information_gain_count"] else None,
            },
        }
    return dict(summary)


def run(args: argparse.Namespace) -> dict:
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"Failure JSONL not found: {input_path}")

    records = []
    source_names = set()
    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            records.append(record)
            source_names.add(record["source_result"])
    if not records:
        raise ValueError(f"No failure records found in {input_path}")

    source_cache: dict[str, dict] = {}
    costs = {
        float(_source_result_for_name(name, source_cache)["training"]["probe_cost"])
        for name in source_names
    }
    if len(costs) != 1:
        raise ValueError("input contains multiple probe costs; analyze each cost separately")
    probe_cost = costs.pop()

    case_rng = random.Random(args.distribution_seed)
    cases = [generate_case(case_rng) for _ in range(args.distribution_episodes)]
    accuracy = _expected_accuracy_by_posterior(cases)
    expected_gains, fold_expected_gains = _expected_net_gain_by_probe(
        accuracy, probe_cost
    )
    candidate_comparisons = (
        ((1 << len(RULES)) - 1)
        * len(ALL_PROBES)
        * (1 + len(FOLD_TEST_RULES))
    )
    alpha = 1.0 - CONFIDENCE_LEVEL
    # Hoeffding + union bound over all posterior/probe pairs and all prior references.
    confidence_half_width = math.sqrt(
        2.0 * math.log(2.0 * candidate_comparisons / alpha)
        / args.distribution_episodes
    )

    groups: dict[tuple[str, int], Counter] = defaultdict(Counter)
    results = []
    for record in records:
        source_result = _source_result_for_name(record["source_result"], source_cache)
        fold = record["fold"]
        if type(fold) is not int or fold not in range(len(FOLD_TEST_RULES)):
            raise ValueError("failure record contains an invalid fold")
        if (
            source_result["fold"] != fold
            or source_result["data"]["test_rule_ids"] != list(FOLD_TEST_RULES[fold])
        ):
            raise ValueError("failure record fold disagrees with its source result metadata")
        source_cost = float(source_result["training"]["probe_cost"])
        if source_cost != probe_cost:
            raise ValueError("probe cost changed while analyzing the input")
        if not math.isfinite(source_cost) or source_cost < 0:
            raise ValueError("source result has an invalid probe cost")
        counterfactual = _analyze_failure(
            record, accuracy, expected_gains, fold_expected_gains,
            probe_cost, confidence_half_width
        )
        results.append({**record, "counterfactual": counterfactual})
        counts = groups[(record["controller"], int(record["budget"]))]
        counts["episodes"] += 1
        counts[counterfactual["case_oracle_outcome"]] += 1
        counts["case_oracle_fixable"] += int(counterfactual["case_oracle_can_find_a_fix"])
        uniform = counterfactual["uniform_candidate_reference"]
        counts[f"uniform_{uniform['query_stop_decision']}"] += 1
        counts["uniform_best_gain"] += uniform["expected_gain_of_best_case_independent_probe"]
        fold_reference = counterfactual["test_fold_conditioned_reference"]
        counts[f"fold_{fold_reference['fold_prior_oracle_best_query_stop_decision']}"] += 1
        counts["fold_best_gain"] += fold_reference[
            "fold_prior_oracle_best_probe_expected_net_gain"
        ]
        if fold_reference["information_gain_probe_expected_net_gain"] is not None:
            counts["information_gain_count"] += 1
            counts["uniform_information_gain"] += uniform[
                "information_gain_probe_expected_net_gain"
            ]
            fold_ig_gain = fold_reference["information_gain_probe_expected_net_gain"]
            counts["fold_information_gain"] += fold_ig_gain
            if fold_ig_gain - confidence_half_width > 0:
                counts["fold_ig_query_supported"] += 1
            elif fold_ig_gain + confidence_half_width < 0:
                counts["fold_ig_stop_supported"] += 1
            else:
                counts["fold_ig_uncertain"] += 1

    if args.output:
        summary_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    else:
        name = input_path.stem.replace("_failures_", "_counterfactual_") + ".json"
        summary_path = input_path.with_name(name)
    details_path = summary_path.with_name(
        summary_path.stem.replace("_counterfactual_", "_counterfactual_cases_") + ".jsonl"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as destination:
        for result in results:
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "experiment": "006B-one-step-counterfactual",
        "failure_records_analyzed": len(records),
        "probe_cost": probe_cost,
        "prior_references": {
            "uniform_candidate_reference": "A reference calculation that treats all currently surviving catalog rules as equally likely; this is not the actual hidden-rule sampling distribution in a held-out fold.",
            "test_fold_conditioned_reference": "Weights hidden rules uniformly over the surviving held-out rules from that fold, matching its test sampling. The policy's answer still uses the full surviving candidate set; only the expected hidden-rule weighting changes. The fold-prior best-query search gives the evaluator fold information the policy itself was not given.",
        },
        "case_distribution": {
            "generator": "generate_case",
            "independent_monte_carlo_cases": args.distribution_episodes,
            "seed": args.distribution_seed,
            "confidence_level": CONFIDENCE_LEVEL,
            "simultaneous_hoeffding_half_width": round(confidence_half_width, 6),
            "simultaneous_comparisons": candidate_comparisons,
        },
        "by_controller_and_budget": _summarize(groups),
        "files": {
            "input_failures": str(input_path),
            "summary": str(summary_path),
            "case_details": str(details_path),
        },
        "interpretation_notes": [
            "All gain averages are conditional on the failure records, not averages over every test episode.",
            "Uniform-candidate expected probe value averages over hidden rules uniformly in the current posterior and over independent cases from the task generator.",
            "Fold-conditioned expected probe value averages only over the surviving held-out rules in that record's fold while preserving the policy's full-candidate answer rule; the fold-prior best probe is an evaluator-side query-selection oracle, while the information-gain probe is chosen from the policy-visible posterior.",
            "A positive net gain means estimated accuracy improvement exceeds the source experiment's probe cost; STOP has zero additional cost.",
            "The simultaneous Hoeffding bound covers all nonempty rule posteriors, probe candidates, and four prior references; it assumes independent case-generator draws.",
            "The case-oracle search sees the particular held-out case before choosing a probe; it is only an optimistic ceiling, not an achievable policy.",
            "One-step results do not rule out a multi-question strategy fixing a case.",
            "For a failure at budget 3, the extra probe is beyond the original evaluation budget.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_controller_and_budget"], indent=2))
    print(f"Hoeffding half-width: {confidence_half_width:.4f} (simultaneous 95% bound)")
    print(f"Failure records analyzed: {len(records)}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate one-extra-probe value on Experiment 006B failures"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--distribution-episodes", type=int, default=32768)
    parser.add_argument("--distribution-seed", type=int, default=CASE_DISTRIBUTION_SEED)
    args = parser.parse_args()
    if args.distribution_episodes <= 0:
        parser.error("--distribution-episodes must be positive")
    run(args)


if __name__ == "__main__":
    main()
