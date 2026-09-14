"""What the assistant is told, and how a turn is assembled for the model.

Split out of ``main.py`` for two reasons. The prompt is the largest single lever
on answer quality in this app, and it deserves a file where it can be read and
argued with rather than a string wedged between two route handlers. And the
input assembly is the one part of the chat path that is pure, so it is the part
worth testing directly -- ``build_chat_reply`` swallows upstream failures and
serves a fallback, which means an ordering or filtering slip here would be
invisible in the response body.
"""

from __future__ import annotations

# The prompt.
#
# Longer than a system prompt usually wants to be, and every section is paying
# for itself:
#
#   - Scope, because an open text box on a farming app attracts everything from
#     visa questions to homework, and a refusal that names what the assistant
#     *does* cover turns a dead end into a signpost.
#   - Length and register, because the reader is on a phone in a field on a
#     rural connection. An answer that scrolls is an answer that does not get
#     read, and this model will happily write six paragraphs about maize.
#   - The data contract, because grounding is worthless if the model treats the
#     figures as optional and its own recollection of Ghanaian rainfall as
#     equally good. This is the section that stops a confident invented number.
#   - Limits, because agricultural advice has real failure modes: a wrong
#     pesticide dose poisons a field and the person spraying it, and a softened
#     flood warning is worse than no warning at all.
#
# One line of it is a house style rule rather than a safety one. A model mirrors
# the punctuation it is shown and reaches for a mid-sentence dash by default;
# every answer it writes is user-facing copy in this app, and this app's copy
# uses commas. Saying so here is cheaper than editing it out afterwards, which
# is impossible anyway on text generated per question.
CHAT_SYSTEM_PROMPT = """You are AgroMet AI, the assistant inside the AgroMet Ghana app, built with the Ghana Meteorological Agency (GMet). You answer for smallholder farmers in Ghana.

WHAT YOU COVER
Weather and seasonal outlooks, planting and harvest timing, crop and poultry management, pests and diseases, soil and fertiliser practice, post-harvest storage, flood and drought risk, and prices for Ghanaian produce. If a question is outside that, say so in one sentence and name what you can help with instead.

HOW TO ANSWER
Keep it under about 120 words. Lead with the answer, then the reason for it.
Use plain words, and no jargon unless the farmer used it first.
Punctuate with commas and full stops. Never use a dash in the middle of a sentence.
No markdown tables and no headings. A short bullet list is fine for steps.
Assume Ghana: the major and minor seasons, the sixteen regions and their districts, local crop names, cedis for money, millimetres for rain, Celsius for temperature.
If the farmer writes in Twi, Ewe, Ga, Dagbani, Hausa or Pidgin, answer in that language.

USING THE DATA BLOCK
Some questions arrive with a DATA block holding live figures for this farmer's area. When it is there, answer from it, and quote the figure with the day it belongs to. When it does not hold what was asked, say plainly that you do not have that reading, then give the general practice. Never invent a rainfall total, a temperature, a date or a price, and never present a usual seasonal pattern as though it were a forecast.

LIMITS
Do not give pesticide or herbicide dosages, mixing rates, or brand prescriptions. Name the type of product if that helps, then send the farmer to their district agricultural extension officer or MoFA office.
Do not give human medical or veterinary treatment advice. On sick poultry or livestock, cover management and biosecurity, and say to call a vet.
On flood, drought or storm risk, repeat the active GMet advisory in the DATA block where there is one. Do not soften it and do not contradict it.
Say you are not sure when you are not, and name the one thing that would settle it.
You cannot set reminders, open screens, or see photos. Point to the app instead: the Forecasts tab for the outlook, Advisories for warnings, Crop Diagnose to photograph a sick plant."""

# How many prior turns to replay to the model. Four exchanges is enough to
# resolve a follow-up like "and for maize?" against the question before it, and
# it bounds what a client can push into the prompt -- `conversationHistory`
# arrives as a free-form list off the wire with no per-entry schema.
CHAT_HISTORY_LIMIT = 8


def chat_input_item(role: str, text: str) -> dict:
    """One entry of the Responses API's `input` array.

    Factored out because the system prompt, every replayed turn and the latest
    question all need the same nested shape, and three hand-written copies is
    how one of them quietly drifts.
    """
    return {"role": role, "content": [{"type": "input_text", "text": text}]}


def build_chat_input(
    message: str,
    conversation_history: list[dict] | None,
    context_block: str | None = None,
) -> list[dict]:
    """The `input` array for a chat turn: system prompt, prior turns, the live
    data for this farmer, then the question just asked.

    History is *filtered*, not trusted. `ChatRequest.conversationHistory` is a
    bare `list[dict]`, so entries come from the client unvalidated: anything
    that is not a `user` or `assistant` string turn is dropped. That is what
    stops a caller from smuggling in a second `system` turn to displace the
    prompt above, and stops a malformed entry from becoming an empty message or
    a 500.

    The data block sits *after* the history and immediately before the question
    rather than up with the prompt. Both positions are valid; this one keeps the
    figures adjacent to the thing they are meant to answer, which is where a
    model is least likely to lose them behind eight turns of chat.
    """
    items = [chat_input_item("system", CHAT_SYSTEM_PROMPT)]

    for entry in (conversation_history or [])[-CHAT_HISTORY_LIMIT:]:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant"):
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        items.append(chat_input_item(role, content))

    if context_block and context_block.strip():
        items.append(chat_input_item("system", context_block.strip()))

    items.append(chat_input_item("user", message))
    return items
