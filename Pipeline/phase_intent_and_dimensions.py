from typing import Any, Dict

from Pipeline.context import PipelineContext
from Pipeline.utils import extract_json_by_key_from_full_text, timed_section


async def run_intent_and_dimensions(ctx: PipelineContext) -> None:
    analyze_intent_runner = ctx.state["analyze_intent_runner"]
    join_column = ctx.state["join_column"]
    decision_log = ctx.state.get("decision_log", {})

    reused_dimensions = ctx.state.get("reuse_dimension_specifications")
    if isinstance(reused_dimensions, dict) and reused_dimensions:
        print("\n↩️ Reusing dimension specifications from round 1.")
        ctx.state["dimension_specifications"] = reused_dimensions
        ctx.pipeline_timings["01_analyze_user_intent_llm"] = 0.0
        ctx.pipeline_timings["02_dimension_interactive_input"] = 0.0
        if isinstance(decision_log, dict):
            decision_log.setdefault("phases", {})["intent_and_dimensions"] = {
                "reused_from_round1": True,
                "dimension_specifications": reused_dimensions,
            }
        return

    print("🚀 Running Table Selection Agent...")
    print(f"📝 User Intent: {ctx.user_intent}")

    analyze_intent_prompt = f"""
User Intent: {ctx.user_intent}

Task Information:
- Target Column: {ctx.target_column}
- Task Type: {ctx.task_type}
- Join Table: {ctx.query_table_display_name}
- Join Columns: {join_column}

Please analyze the user intent and return the result in JSON format according to the prompt.
"""
    with timed_section(ctx.pipeline_timings, "01_analyze_user_intent_llm"):
        events = await analyze_intent_runner.run_debug(analyze_intent_prompt, quiet=True)
        last_text = ""
        for event in events:
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                for part in event.content.parts:
                    t = getattr(part, "text", None)
                    if t:
                        last_text = t
        # Prefer key-aware extraction: greedy extract_json() often matches CoT
        # fragments and returns {} / wrong objects, which skips all interactive
        # confirmations and leaves search_query empty.
        analyzed_intent = None
        if "domain_field" in (last_text or ""):
            analyzed_intent = extract_json_by_key_from_full_text(
                last_text, "domain_field", prefer_non_empty_list=False
            )
            if not isinstance(analyzed_intent, dict) or not isinstance(
                analyzed_intent.get("domain_field"), dict
            ):
                analyzed_intent = None
        if analyzed_intent is None:
            print(
                "[warn] analyze_user_intent JSON parse failed; "
                "falling back to interactive prompts for all dimensions"
            )
            analyzed_intent = {
                "domain_field": {"is_explicitly_mentioned": False},
                "geographic": {"is_explicitly_mentioned": False},
                "temporal": {"is_explicitly_mentioned": False},
                "population_group": {"is_explicitly_mentioned": False},
            }

    dimension_specifications: Dict[str, Any] = {}
    with timed_section(ctx.pipeline_timings, "02_dimension_interactive_input"):
        for dim_key, dim_display_name in [
            ("domain_field", "Domain/Field"),
            ("geographic", "Geographic"),
            ("temporal", "Temporal"),
            ("population_group", "Population Group"),
        ]:
            dim_info = analyzed_intent.get(dim_key) if analyzed_intent else None
            if not isinstance(dim_info, dict):
                continue
            is_explicitly_mentioned = dim_info.get("is_explicitly_mentioned") is True
            explicitly_mentioned_value = dim_info.get("explicitly_mentioned_value")
            if is_explicitly_mentioned:
                raw = explicitly_mentioned_value
                value_str = ", ".join(str(x) for x in raw) if isinstance(raw, list) else str(raw or "")
                print(f"\nDimension '{dim_display_name}' is set to: {value_str}")
                print("Reply 'done' to confirm, or type the correct value.")
                user_input = input("Your reply: ").strip()
                confirmed_value = explicitly_mentioned_value if user_input.lower() == "done" else user_input
                if isinstance(confirmed_value, list):
                    dimension_specifications[dim_display_name] = confirmed_value
                else:
                    dimension_specifications[dim_display_name] = [confirmed_value]
            else:
                suggested = dim_info.get("suggested_values") or []
                sug_str = f" Suggested: {', '.join(str(x) for x in suggested)}." if suggested else ""
                print(f"\nDimension '{dim_display_name}' was not specified in your intent.")
                print(f"Enter a value{sug_str} or type 'skip' to skip.")
                user_input = input("Your reply: ").strip()
                if user_input.lower() != "skip" and user_input:
                    dimension_specifications[dim_display_name] = [user_input]
                else:
                    dimension_specifications[dim_display_name] = ["all"]
        print("=" * 40, "Dimension Specifications", "=" * 40)
        print(dimension_specifications)

    ctx.state["dimension_specifications"] = dimension_specifications
    if isinstance(decision_log, dict):
        decision_log.setdefault("phases", {})["intent_and_dimensions"] = {
            "reused_from_round1": False,
            "dimension_specifications": dimension_specifications,
        }

