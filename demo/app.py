from typing import Dict, List, Optional, Tuple

import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_ENTITY_TYPES = [
    "PLAYER", "TEAM", "POSITION", "STAT", "INJURY",
    "TRADE_ASSET", "GAME_EVENT", "VENUE", "COACH", "AWARD",
]

EXAMPLE_INPUTS = [
    (
        "LeBron James scored 32 points and dished out 11 assists as the Los Angeles Lakers "
        "defeated the Golden State Warriors 118-105 at Crypto.com Arena. Head coach Darvin Ham "
        "praised James's leadership following the win.",
        ALL_ENTITY_TYPES,
    ),
    (
        "Patrick Mahomes threw for 312 yards and three touchdowns, leading the Kansas City Chiefs "
        "to a 27-17 victory over the Buffalo Bills at Arrowhead Stadium. Tight end Travis Kelce "
        "was listed as questionable with an ankle injury.",
        ALL_ENTITY_TYPES,
    ),
    (
        "The New York Yankees acquired outfielder Juan Soto from the San Diego Padres in a "
        "blockbuster trade, sending four prospects including a first-round draft pick to San Diego. "
        "Soto, batting .275 with 35 home runs, will join slugger Aaron Judge in the Yankees lineup.",
        ALL_ENTITY_TYPES,
    ),
]

LABEL_COLORS: Dict[str, str] = {
    "PLAYER":       "#dbeafe",
    "TEAM":         "#dcfce7",
    "POSITION":     "#fef9c3",
    "STAT":         "#fce7f3",
    "INJURY":       "#fee2e2",
    "TRADE_ASSET":  "#ede9fe",
    "GAME_EVENT":   "#ffedd5",
    "VENUE":        "#e0f2fe",
    "COACH":        "#d1fae5",
    "AWARD":        "#fef3c7",
}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str = "checkpoints/gliner-sports"):
    """
    Load the fine-tuned GLiNER model for serving.

    Args:
        model_path: Path to the fine-tuned checkpoint directory.

    Returns:
        Loaded GLiNER model instance.
    """
    # TODO: Import GLiNER and call GLiNER.from_pretrained(model_path).
    raise NotImplementedError


def run_extraction(
    text: str,
    entity_types: List[str],
    threshold: float,
    model,
) -> List[Dict]:
    """
    Run entity extraction on the input text.

    Args:
        text: Raw sports text.
        entity_types: List of entity type labels to extract.
        threshold: Confidence threshold.
        model: Loaded GLiNER model.

    Returns:
        List of entity dicts with keys: text, label, start, end, score.
    """
    # TODO: Call model.predict_entities(text, entity_types, threshold=threshold).
    # TODO: Return the resulting entity list.
    raise NotImplementedError


def highlight_entities(text: str, entities: List[Dict]) -> List[Tuple]:
    """
    Convert entity predictions to the Gradio HighlightedText format.

    Args:
        text: Original input text.
        entities: List of entity dicts from run_extraction.

    Returns:
        List of (text_segment, label_or_None) tuples for gr.HighlightedText.
    """
    # TODO: Sort entities by start offset.
    # TODO: Build segments by interleaving plain spans and labelled spans.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Tab: Extraction
# ---------------------------------------------------------------------------

def extraction_tab(model) -> gr.Tab:
    """Build the entity extraction demo tab."""
    with gr.Tab("Entity Extraction") as tab:
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Sports Article Text",
                    placeholder="Paste a sports article or news snippet here…",
                    lines=8,
                )
                entity_selector = gr.CheckboxGroup(
                    choices=ALL_ENTITY_TYPES,
                    value=ALL_ENTITY_TYPES,
                    label="Entity Types",
                )
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                )
                extract_btn = gr.Button("Extract Entities", variant="primary")

            with gr.Column(scale=3):
                highlighted_output = gr.HighlightedText(
                    label="Extracted Entities",
                    combine_adjacent=False,
                    color_map=LABEL_COLORS,
                )
                entity_table = gr.Dataframe(
                    headers=["Entity", "Label", "Start", "End", "Score"],
                    label="Entity Details",
                )

        gr.Examples(
            examples=[[ex[0], ex[1]] for ex in EXAMPLE_INPUTS],
            inputs=[text_input, entity_selector],
            label="Example Inputs",
        )

        # TODO: Wire extract_btn.click → run_extraction → highlight_entities + entity_table.
    return tab


# ---------------------------------------------------------------------------
# Tab: Benchmark
# ---------------------------------------------------------------------------

def benchmark_tab() -> gr.Tab:
    """Build the benchmark comparison stub tab."""
    with gr.Tab("Benchmark Comparison") as tab:
        gr.Markdown("## System Comparison: Base GLiNER vs Fine-Tuned GLiNER vs GPT-4o-mini")
        benchmark_table = gr.Dataframe(
            headers=["System", "F1", "Precision", "Recall", "P50 Latency (ms)", "P99 Latency (ms)", "Cost / 1K docs"],
            label="Benchmark Results",
        )
        refresh_btn = gr.Button("Load Latest Benchmark Results")

        # TODO: Wire refresh_btn.click to load benchmark JSON and populate benchmark_table.
    return tab


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble and return the full Gradio Blocks application."""
    # TODO: Call load_model() once at startup and pass to extraction_tab.
    model = None  # placeholder until load_model is implemented

    with gr.Blocks(title="GLiNER Sports NER Demo") as demo:
        gr.Markdown("# GLiNER Sports NER Demo")
        gr.Markdown(
            "Fine-tuned named entity recognition for sports articles. "
            "Extracts players, teams, stats, injuries, and more."
        )
        extraction_tab(model)
        benchmark_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
