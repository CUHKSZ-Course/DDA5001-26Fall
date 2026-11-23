"""Two guided comparisons on paired evaluation records."""


def math_by_level(questions):
    """Group math questions by difficulty.

    Each question has a level and correct_draws for original, teacher, sft and opd.
    Return a dict keyed by the levels present. Each value contains questions,
    pass_at_1 (a dict of fractions for all four arms), sft_wins, opd_wins and ties.
    Count a question in sft_wins or opd_wins when that method has more correct draws.
    """
    # TODO 1: group questions by level. Aggregate scores and compare the two
    # adapters on each question before counting wins and ties.
    raise NotImplementedError("Complete math_by_level in analysis_exercises.py")


def ifeval_transitions(questions):
    """Compare each adapter with the original student at the pass@4 threshold.

    Return a dict keyed by sft and opd. For each adapter, count retained
    (pass to pass), lost (pass to fail), gained (fail to pass) and
    still_failing (fail to fail). A question passes if at least one draw succeeds.
    """
    # TODO 2: classify each question for each adapter. Each question belongs
    # to exactly one category, even if several of its draws pass.
    raise NotImplementedError("Complete ifeval_transitions in analysis_exercises.py")
