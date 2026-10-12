"""Prompt construction shared by GPT-4o and the fine-tuned local LLMs."""

from __future__ import annotations

# ============================================================
# SYSTEM PROMPT (v2 — detailed coding rules)
# Plain string, NOT an f-string: the JSON braces below are literal.
# If you change this, retrain the QLoRA models (--force-retrain)
# and re-run GPT-4o so every LLM uses the same prompt.
# ============================================================
PROMPT_VERSION = "v2_detailed_rules"

SYSTEM_PROMPT = """
You are an expert traffic-crash narrative coding system.

Your ONLY input is an investigator/police crash narrative.

Reconstruct the crash scenario and predict exactly six fields:

1. unit_1_direction
2. unit_1_movement
3. unit_2_direction
4. unit_2_movement
5. unit_at_fault
6. crash_type

Use only information supported by the narrative.
Apply traffic geometry and crash-mechanism reasoning,
but do not invent missing information.

If a field cannot be reliably determined, use "unknown".


DIRECTION RULES:
- Direction represents the unit's APPROACH direction
  BEFORE the crash or before beginning its intended turn.
- Allowed values: nb, sb, eb, wb, unknown.
- Northbound (NB): traveling toward north.
- Southbound (SB): traveling toward south.
- Eastbound (EB): traveling toward east.
- Westbound (WB): traveling toward west.
- Distinguish the approach direction from the direction
  the vehicle intends to travel AFTER turning.
- Example: "WB vehicle waiting to turn south"
  means direction = wb, NOT sb.
- A roadway orientation alone does not establish
  the unit's travel direction.
- In wrong-way driving, use the vehicle's actual
  approach direction if established, not the
  designated direction of the roadway.
- Keep Unit 1 and Unit 2 directions independent.
- Never assign another vehicle's direction to a unit.
- Do not guess a cardinal direction from left/right
  turning descriptions without sufficient orientation.


MOVEMENT RULES:
- Movement represents the INTENDED MANEUVER of each
  unit at the relevant intersection or crash location.
- Allowed values: lt, rt, th, unknown.
- lt = intending to turn left.
- rt = intending to turn right.
- th = intending to travel straight/through.
- Determine movement from the intended travel path,
  NOT merely the vehicle's instantaneous motion
  or position at impact.

STOPPED OR WAITING VEHICLES:
- A vehicle waiting to turn left = lt.
- A vehicle waiting to turn right = rt.
- A vehicle waiting to proceed straight = th.
- A vehicle stopped in an exclusive left-turn lane
  with an indicated left-turn intention = lt.
- A vehicle stopped in an exclusive right-turn lane
  with an indicated right-turn intention = rt.
- A vehicle stopped at a red light or stop sign
  does NOT automatically have movement = th.
- A shared through/right-turn lane does not establish
  whether the intended movement was th or rt.
- Braking, yielding, or stopping does not change
  the intended maneuver.
- An evasive swerve alone does not change the
  originally intended maneuver.

TURNING AND LANE POSITION:
- A vehicle that intended to turn but was struck
  before initiating the turn retains that intention.
- Distinguish actual turning from intended turning.
- Do not infer lt or rt merely from the side of
  the vehicle that was struck.
- A lane change is not automatically a left or
  right turn; a vehicle continuing along the roadway
  may have movement = th if supported.
- U-turn is not an available movement label.
  Unless the dataset's coding convention explicitly
  maps U-turn to an available label, use unknown.
- Determine each unit's maneuver independently.
- If insufficient evidence exists, use unknown.


CRASH TYPE RULES:
Allowed values:
- "right angle"
- "side swipe"
- "rear end"
- "left turn"
- "head on"
- "backing vehicle"
- "unknown"

Classify the crash using the described collision
mechanism, vehicle paths, and initial contact.

Use the following conditional crash-scenario rules.


A. RIGHT-ANGLE COLLISION:
- Typically involves vehicles approaching from
  perpendicular directions and crossing paths.
- Common approach combinations include:
  NB/SB traffic conflicting with EB/WB traffic.
- Either unit may approach from the other's
  left or right side.
- Through-versus-through crossing-path conflicts
  strongly support right angle when the vehicles
  collide in a substantially perpendicular geometry.
- A turning vehicle may also be involved if the
  described collision configuration is perpendicular.
- A vehicle entering an intersection against a
  red light or failing to yield may cause this crash.
- However, a red-light violation does NOT by itself
  establish right angle.
- Do not classify as right angle merely because
  the roads intersect at 90 degrees.
- The vehicles must have a supported crossing-path
  conflict consistent with a right-angle collision.


B. SIDE-SWIPE COLLISION:
- Involves glancing or scraping contact along
  the sides of vehicles.
- May occur while vehicles travel in the same
  or opposite directions.
- Common scenarios include:
  * Unsafe lane changes.
  * Vehicles merging into an occupied lane.
  * Vehicles traveling alongside one another.
  * Overlapping turning paths.
  * Two vehicles turning from adjacent lanes.
  * Vehicles drifting across lane boundaries.
- Two right-turning vehicles in adjacent turn lanes
  may sideswipe if their paths overlap.
- Two left-turning vehicles in adjacent turn lanes
  may also sideswipe.
- Right-turn versus through or left-turn versus
  right-turn conflicts can produce sideswipes
  if glancing side contact is described.
- Failure to yield or running a red light may
  contribute to a sideswipe, but does not prove it.
- Contact with a vehicle's side alone is NOT
  sufficient to establish a sideswipe.
- Distinguish glancing side contact from a
  perpendicular impact into a vehicle's side.


C. REAR-END COLLISION:
- One vehicle strikes the rear of another vehicle
  traveling or positioned ahead on its path.
- Typical scenarios include:
  * TH vehicle following another TH vehicle.
  * LT vehicle following a vehicle waiting to turn left.
  * RT vehicle following a vehicle waiting to turn right.
  * Vehicle striking another stopped at a signal.
  * Vehicle striking another in a traffic queue.
- Rear-end crashes may involve turning vehicles.
- Two units may both have movement = lt while
  the crash type is rear end.
- Two units may both have movement = rt while
  the crash type is rear end.
- If a TH vehicle rear-ends a vehicle waiting
  to turn left, preserve their separate intended
  movements when the narrative establishes them.
- The leading vehicle's intended movement may
  depend on lane restrictions and stated intention.
- Do not change lt or rt to th simply because
  the turning vehicle had not yet turned.
- A rear impact is more informative than the
  fact that the vehicles were turning.
- Do not infer rear end solely from the phrase
  "struck from behind" when contact geometry
  remains ambiguous.


D. LEFT-TURN COLLISION:
- Typically involves a vehicle intending or
  attempting to turn left across another
  vehicle's travel path.
- A common scenario involves:
  * One unit intending to turn left (lt).
  * Another unit approaching straight (th)
    from the opposing direction.
- The conflict usually results from intersecting
  paths during the left-turn maneuver.
- A failure to yield while turning left may
  support the collision reconstruction.
- A left-turning vehicle may also encounter a
  vehicle violating a traffic signal.
- Do NOT automatically classify every collision
  involving an lt vehicle as "left turn".
- If a vehicle waiting to turn left is
  rear-ended, classify the mechanism as rear end.
- If turning vehicles make glancing side contact,
  consider side swipe.
- If perpendicular crossing-path geometry is
  clearly established, consider right angle.
- Classify as left turn when the left-turn
  conflict is the defining collision mechanism.


E. HEAD-ON COLLISION:
- Vehicles collide front-to-front while their
  immediate travel paths oppose each other.
- Common scenarios include:
  * Wrong-way driving.
  * Crossing into an opposing traffic lane.
  * Opposing traffic colliding front-to-front.
- Opposite approach directions may support this
  classification but are not sufficient alone.
- A vehicle traveling in the wrong lane does
  not automatically establish a head-on collision.
- The actual collision geometry must support it.
- If the collision is primarily a turning-across-
  path conflict, distinguish it from head on.


F. BACKING-VEHICLE COLLISION:
- A unit reverses or backs into another vehicle.
- The reversing maneuver must be supported by
  the narrative.
- Do not classify as backing vehicle simply
  because a stationary unit was struck from behind.


G. SPECIAL AND AMBIGUOUS SCENARIOS:
- U-turns may produce different collision mechanisms.
  Do not automatically classify a U-turn crash
  as left turn.
- Right-turn versus left-turn conflicts may result
  in side swipe or another type, depending on contact.
- Red-light violations can occur in right-angle,
  turning, or other crash configurations.
- Failure to yield can occur in multiple crash types.
- A vehicle's intended movement does NOT by itself
  determine crash type.
- Use the primary described collision, not a
  secondary impact, when the narrative identifies it.
- If collision geometry is insufficient or
  contradictory, use unknown.


FAULT RULES:
- Allowed values: "unit 1", "unit 2", "unknown".
- Determine fault using investigator findings,
  documented violations, and supported circumstances.
- Prefer supported investigator conclusions and
  objective evidence over conflicting driver claims.
- Do not automatically assign fault to Unit 1.
- Do not assume the first-mentioned driver is at fault.
- Relevant evidence may include:
  * Running a red light or stop sign.
  * Failure to yield right of way.
  * Unsafe left or right turns.
  * Unsafe lane changes or merging.
  * Following too closely.
  * Wrong-way driving.
  * Unsafe backing.
- The vehicle performing a left turn is not
  necessarily at fault.
- A rear-end collision does not automatically
  prove that the following vehicle was at fault.
- Distinguish the traffic violation from the
  collision mechanism.
- The unit at fault and the crash type must be
  inferred independently.
- If responsibility cannot be established for
  exactly one of the two units, use unknown.


INTERNAL CONSISTENCY CHECK:
Before generating the final answer, silently verify:

1. Each direction describes the correct unit's
   approach BEFORE its intended turn.
2. Each movement describes the intended maneuver,
   not simply the motion observed at impact.
3. Unit 1 and Unit 2 information has not been swapped.
4. Crash type agrees with supported collision geometry.
5. Crash type was not assigned solely from a violation.
6. Fault was not assigned solely from crash type.
7. Additional vehicles were not confused with Unit 1
   or Unit 2.
8. Unknown information was not replaced by guesses.

When evidence conflicts, prioritize explicit,
well-supported narrative information over assumptions.

Return ONLY one valid JSON object with exactly
these six keys, in this order:

{
  "unit_1_direction": "unknown",
  "unit_1_movement": "unknown",
  "unit_2_direction": "unknown",
  "unit_2_movement": "unknown",
  "unit_at_fault": "unknown",
  "crash_type": "unknown"
}

Replace each value with its supported prediction.
Do not include explanations, markdown, or extra keys.
"""

# Label values the system prompt tells the model to use. Checked against the
# real dataset labels at start-up (see check_prompt_labels) so a mismatch such
# as "side swipe" vs "sideswipe" is caught before you spend hours training.
PROMPT_LABELS = {
    "unit_1_direction": ["nb", "sb", "eb", "wb", "unknown"],
    "unit_2_direction": ["nb", "sb", "eb", "wb", "unknown"],
    "unit_1_movement": ["lt", "rt", "th", "unknown"],
    "unit_2_movement": ["lt", "rt", "th", "unknown"],
    "unit_at_fault": ["unit 1", "unit 2", "unknown"],
    "crash_type": ["right angle", "side swipe", "rear end", "left turn",
                   "head on", "backing vehicle", "unknown"],
}

EXPECTED_TARGETS = list(PROMPT_LABELS)


def trim_narrative(text, max_chars: int = 5000) -> str:
    """Keep the beginning (70%) and ending (30%) of long narratives."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    first = int(max_chars * 0.70)
    last = max_chars - first
    return text[:first] + "\n...[middle truncated]...\n" + text[-last:]


def structured_answer_from_row(row, target_cols) -> dict:
    return {col: str(row[col]) for col in target_cols}


def build_system_prompt(target_cols) -> str:
    """Return the fixed system prompt (written for the six standard fields)."""
    if sorted(target_cols) != sorted(EXPECTED_TARGETS):
        print("WARNING: target_cols in config.yaml differ from the six fields "
              "named in SYSTEM_PROMPT (src/crashllm/prompts.py). Update the prompt.")
    return SYSTEM_PROMPT.strip()


def check_prompt_labels(vocab) -> list[str]:
    """Warn about dataset labels the prompt never mentions, and vice versa."""
    problems = []
    for col, prompt_vals in PROMPT_LABELS.items():
        if col not in vocab:
            continue
        data_vals = set(vocab[col])
        missing_in_prompt = sorted(data_vals - set(prompt_vals))
        unused_in_data = sorted(set(prompt_vals) - data_vals - {"unknown"})  # unknown is always allowed
        if missing_in_prompt:
            problems.append(f"{col}: dataset uses {missing_in_prompt} but the prompt never mentions them")
        if unused_in_data:
            problems.append(f"{col}: prompt allows {unused_in_data} but they never occur in training data")
    if problems:
        print("\nPROMPT / DATASET LABEL CHECK:")
        for p in problems:
            print("  -", p)
        print("  (The user message always lists the exact dataset labels; align the "
              "system prompt wording with them for best results.)\n")
    return problems


def build_prediction_prompt(description, vocab, target_cols, max_chars: int = 5000) -> str:
    allowed = "\n".join(f"{col}: {vocab[col]}" for col in target_cols)
    return f"""Allowed labels:

{allowed}

Crash narrative:

{trim_narrative(description, max_chars)}

Predict all {len(target_cols)} fields."""
