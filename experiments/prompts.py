"""Paper Fig 5 reconstruction — zero-shot emotion classification prompt
for Llama-3.1-Instruct. Structure mirrors
``data_generation/labeling/labeling_gpt.py`` (the GPT-4o-mini labeling prompt
used to build emoprism.json.gz) adapted for decoder-only chat format.

RQ mapping: shared by all three research questions — every forward pass
during neuron selection (RQ1), masking evaluation (RQ2), and the
ratio/layer sweep (RQ3) formats inputs via :func:`format_messages`.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a sentiment analysis expert."

USER_PROMPT_TEMPLATE = """\
Analyze the following conversation and determine the primary emotion that \
best represents the overall sentiment expressed throughout the dialogue. \
Select only from the following emotions: anger, disgust, fear, happiness, sadness, surprise.

Conversation:
{dialogue}

Please respond with only the emotion name from the provided list. \
Do not include any additional text, formatting, or explanations."""

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Regression fixtures — spot-check that the prompt elicits a single emotion word.
# Expected behavior on Llama-3.1-8B-Instruct (paper's target model):
FIXTURES = [
    # (dialogue, expected_emotion)
    (
        "A: I can't believe we got into Stanford!\n"
        "B: Me too, I'm so thrilled. This is the best day.",
        "happiness",
    ),
    (
        "A: This stew is burnt beyond recognition!\n"
        "B: Don't be dramatic, it's fine.\n"
        "A: Fine?! This is a culinary disaster!",
        "anger",
    ),
    (
        "A: They're launching another mission to Mars already?\n"
        "B: Wait, seriously? I thought we were years away.\n"
        "A: I know! I'm flabbergasted.",
        "surprise",
    ),
]


def format_messages(dialogue: str) -> list[dict]:
    """Return chat-template messages for Llama-3.1-Instruct.

    The returned list is intended to be passed to
    ``tokenizer.apply_chat_template(..., add_generation_prompt=True)``
    so the model is primed to emit its single-word answer on the
    assistant turn.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(dialogue=dialogue)},
    ]


if __name__ == "__main__":
    # Smoke print: show a formatted fixture so callers can sanity-check the
    # prompt structure without loading any model weights.
    dialogue, expected = FIXTURES[0]
    msgs = format_messages(dialogue)
    print(f"[fixture expected={expected}]")
    for m in msgs:
        print(f"--- {m['role']} ---")
        print(m["content"])
