import pandas as pd

from rofa.papers.from_answers_to_hypotheses import analysis


def test_top2_flip_analysis_strict_mode() -> None:
    patterns = [
        analysis.VotePatternSpec(
            top1_votes=8, top2_votes=2, examples=5, top1_correct=4, top2_correct=1
        ),
        analysis.VotePatternSpec(
            top1_votes=7, top2_votes=3, examples=6, top1_correct=3, top2_correct=2
        ),
        analysis.VotePatternSpec(
            top1_votes=6, top2_votes=3, examples=4, top1_correct=1, top2_correct=2
        ),
        analysis.VotePatternSpec(
            top1_votes=6, top2_votes=2, examples=4, top1_correct=0, top2_correct=0, tie_top2=4
        ),
    ]
    df = analysis.generate_synthetic_vote_pattern_dataset(patterns, n_branches=10, seed=7)
    matrix, rectangles, threshold_rectangles, tie_stats = analysis.top2_flip_analysis(
        df, strict=True, min_support=1, ratio_thresholds=(2.0,)
    )

    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] == 90
    assert tie_stats["included_examples"] == 15
    assert tie_stats["excluded_tie_top2"] == 4
    assert int(matrix["examples_count"].sum()) == 15

    cell_8_2 = matrix[(matrix["top1_votes"] == 8) & (matrix["top2_votes"] == 2)].iloc[0]
    assert cell_8_2["examples_count"] == 5
    assert cell_8_2["top1_correct_count"] == 4
    assert cell_8_2["top2_correct_count"] == 1

    rect = rectangles[
        (rectangles["top1_votes_min"] == 6)
        & (rectangles["top1_votes_max"] == 8)
        & (rectangles["top2_votes_min"] == 2)
        & (rectangles["top2_votes_max"] == 3)
    ].iloc[0]
    assert rect["total_examples_count"] == 15
    assert rect["total_top1_correct_count"] == 8
    assert rect["total_top2_correct_count"] == 5
    assert threshold_rectangles.shape[0] == 1
