"""Small CPU checks for the paired analysis exercises."""

from analysis_exercises import ifeval_transitions, math_by_level


def question(question_id, level, original, teacher, sft, opd):
    return {
        "id": question_id,
        "level": level,
        "correct_draws": {"original": original, "teacher": teacher, "sft": sft, "opd": opd},
    }


def main():
    math = [
        question("a", 1, 3, 4, 3, 2),
        question("b", 5, 0, 4, 1, 3),
        question("c", 1, 1, 3, 2, 2),
    ]
    stats = math_by_level(math)
    assert stats == {
        1: {
            "questions": 2,
            "pass_at_1": {"original": 0.5, "teacher": 0.875, "sft": 0.625, "opd": 0.5},
            "sft_wins": 1, "opd_wins": 0, "ties": 1,
        },
        5: {
            "questions": 1,
            "pass_at_1": {"original": 0.0, "teacher": 1.0, "sft": 0.25, "opd": 0.75},
            "sft_wins": 0, "opd_wins": 1, "ties": 0,
        },
    }, "Group by the supplied levels and compare the number of successful draws per question"

    instructions = [
        question("a", None, 3, 4, 1, 0),
        question("b", None, 1, 4, 0, 4),
        question("c", None, 0, 4, 3, 1),
        question("d", None, 0, 3, 0, 1),
    ]
    assert ifeval_transitions(instructions) == {
        "sft": {"retained": 1, "lost": 1, "gained": 1, "still_failing": 1},
        "opd": {"retained": 1, "lost": 1, "gained": 2, "still_failing": 0},
    }, "Count once per question, with at least one successful draw defining pass@4 success"
    print("Both analysis checks passed.")


if __name__ == "__main__":
    main()
