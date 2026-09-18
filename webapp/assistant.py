"""Claude API integration: open-ended conversation plus tool-calling so the
model can actually operate the MLP trainer when the user asks it to.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import anthropic

MODEL = "claude-sonnet-5"
MAX_TOKENS = 1024
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """\
Je bent de JAX MLP Assistent: een vriendelijke, behulpzame AI die net zo vrij \
en natuurlijk converseert als ChatGPT of Claude normaal doen, over eender welk \
onderwerp. Je bent tegelijk gekoppeld aan een echt lokaal JAX/Flax MLP-trainings\
raamwerk via een paar tools.

Gebruik de tools wanneer de gebruiker vraagt om een model te trainen, een \
voorspelling te doen, de modelarchitectuur te tonen, of de trainingsstatus op \
te vragen - of wanneer dat duidelijk hun bedoeling is, ook als ze het niet \
letterlijk zo formuleren. Kies verstandige standaardwaarden voor hyperparameters \
die niet genoemd zijn; vraag alleen door bij echte onduidelijkheid.

Bij regressie leert het model altijd dezelfde synthetische formule: \
y = 2*x1 + 3*x2 + ruis, met de rest van de features als irrelevante ruis-kolommen. \
Bij classificatie zijn de klassen willekeurig gegenereerd (niet betekenisvol \
interpreteerbaar), dus leg dat kort uit als de gebruiker ernaar vraagt.

Antwoord bondig maar warm, in het Nederlands, tenzij de gebruiker een andere \
taal gebruikt - volg dan die taal. Voor alles buiten de trainer (algemene \
vragen, uitleg, grapjes, advies) gedraag je je gewoon als een volwaardige, \
behulpzame chatbot."""

TOOLS = [
    {
        "name": "start_training",
        "description": (
            "Start het trainen van een nieuw MLP-model op synthetische data "
            "(regressie of classificatie). Draait op de achtergrond; de "
            "gebruiker ziet live voortgang in de UI."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "enum": ["regression", "classification"],
                    "description": "Type taak.",
                },
                "num_classes": {
                    "type": "integer",
                    "description": "Aantal klassen, alleen bij classification (standaard 4).",
                },
                "input_dim": {
                    "type": "integer",
                    "description": "Aantal invoerfeatures (standaard 4).",
                },
                "hidden_dim": {
                    "type": "integer",
                    "description": "Breedte van elke hidden layer (standaard 64).",
                },
                "num_layers": {
                    "type": "integer",
                    "description": "Aantal hidden layers (standaard 2).",
                },
                "dropout": {
                    "type": "number",
                    "description": "Dropout-kans, 0-0.9 (standaard 0.1).",
                },
                "num_steps": {
                    "type": "integer",
                    "description": "Aantal trainingsstappen (standaard 300).",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batchgrootte (standaard 32).",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate (standaard 0.001).",
                },
                "num_samples": {
                    "type": "integer",
                    "description": "Aantal gegenereerde samples (standaard 1000).",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed (standaard 42).",
                },
            },
            "required": ["task"],
        },
    },
    {
        "name": "predict",
        "description": (
            "Doe een voorspelling met het meest recent getrainde model. "
            "'features' moet exact evenveel getallen bevatten als de "
            "input-dimensie waarmee het model getraind is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Featurewaarden x1..xN.",
                }
            },
            "required": ["features"],
        },
    },
    {
        "name": "get_model_summary",
        "description": (
            "Haal de architectuur en parameteraantallen op van het huidige getrainde model."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_training_status",
        "description": (
            "Vraag de huidige trainingsstatus en voortgang op (idle/training/done/error)."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

ToolExecutor = Callable[
    [str, Dict[str, Any]], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]
]


class AssistantError(Exception):
    """Raised when the Claude API can't be reached or is misconfigured."""


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise AssistantError(
            "Geen ANTHROPIC_API_KEY geconfigureerd. Maak een .env bestand aan in de "
            "projectroot met de regel ANTHROPIC_API_KEY=sk-ant-... "
            "(zie .env.example)."
        )
    return anthropic.Anthropic(api_key=api_key)


def chat_turn(
    history: List[Dict[str, str]],
    user_message: str,
    execute_tool: ToolExecutor,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Runs one full conversational turn, including any tool calls.

    Args:
        history: Prior turns as [{"role": "user"|"assistant", "content": str}, ...].
        user_message: The new message from the user.
        execute_tool: Callback (tool_name, tool_input) -> (result_for_claude,
            ui_action). ``ui_action`` (if not None) is surfaced to the caller
            so the frontend can render rich content (e.g. a live training
            card) alongside Claude's natural-language reply.

    Returns:
        A tuple ``(reply_text, ui_action)``.
    """
    client = _client()

    messages: List[Dict[str, Any]] = [
        {"role": h["role"], "content": h["content"]} for h in history
    ]
    messages.append({"role": "user", "content": user_message})

    ui_action: Optional[Dict[str, Any]] = None

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as exc:
            raise AssistantError(
                f"Fout bij aanroepen van de Claude API: {exc}"
            ) from exc

        if response.stop_reason != "tool_use":
            text_parts = [
                block.text for block in response.content if block.type == "text"
            ]
            return "".join(text_parts).strip() or "(geen antwoord)", ui_action

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result, action = execute_tool(block.name, block.input or {})
            if action is not None:
                ui_action = action
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return (
        "Ik ben te lang bezig geweest met tools aanroepen en stop hier. Probeer het opnieuw of "
        "formuleer je vraag iets anders.",
        ui_action,
    )
