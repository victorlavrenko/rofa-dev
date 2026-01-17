"""Token helpers shared across generation utilities."""

from __future__ import annotations

from typing import List


def get_eos_ids(tokenizer) -> List[int]:
    """Return the list of EOS token ids, including <|eot_id|> when present."""
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            eos_ids.append(eot)
    except Exception:  # noqa: BLE001
        pass
    return list({i for i in eos_ids if isinstance(i, int)})
