# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rlinf.models.embodiment.uninavid.nav_rollout import (
    build_action_token_mask,
    count_parsed_action_chars,
    count_response_alpha_chars,
)


def test_count_parsed_action_chars_counts_only_parsed_action_words():
    assert count_parsed_action_chars("forward left right stop", 4) == 20


def test_effective_word_ratio_helpers_count_numerator_and_denominator():
    output_text = "forward, then left. 123!"

    assert count_parsed_action_chars(output_text, 4) == 11
    assert count_response_alpha_chars(output_text) == 15


def test_count_parsed_action_chars_stops_at_stop_action():
    assert count_parsed_action_chars("forward stop left right", 4) == 11


def test_count_parsed_action_chars_stops_at_num_action_chunks():
    assert count_parsed_action_chars("forward left right stop", 2) == 11


def test_count_parsed_action_chars_does_not_count_padded_no_op():
    assert count_parsed_action_chars("left", 4) == 4


def test_build_action_token_mask_marks_parsed_action_tokens_only():
    response_ids = [6375, 29892, 2175, 29892, 1492, 29892, 5040, 29889]

    mask = build_action_token_mask("forward, left, right, stop.", response_ids, 4)

    assert mask == [True, False, True, False, True, False, True, False]


def test_build_action_token_mask_handles_boundary_action_token_variants():
    response_ids = [6375, 13, 1563, 13, 1266, 13, 9847]

    mask = build_action_token_mask("forward\nleft\nright\nstop", response_ids, 4)

    assert mask == [True, False, True, False, True, False, True]


def test_build_action_token_mask_does_not_mark_word_prefix_matches():
    response_ids = [6375, 2264, 1492, 1319, 5040, 957]

    mask = build_action_token_mask("forwardness rightful stopover", response_ids, 4)

    assert mask == [False, False, False, False, False, False]
