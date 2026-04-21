import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd


import os
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

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
    "PLAYER":       "#2563eb",  # blue
    "TEAM":         "#16a34a",  # green
    "POSITION":     "#ca8a04",  # yellow-dark
    "STAT":         "#db2777",  # pink-dark
    "INJURY":       "#dc2626",  # red
    "TRADE_ASSET":  "#7c3aed",  # purple
    "GAME_EVENT":   "#ea580c",  # orange
    "VENUE":        "#0891b2",  # cyan-dark
    "COACH":        "#059669",  # emerald
    "AWARD":        "#d97706",  # amber
}

BENCHMARK_PATH = Path("data/benchmark_results.json")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str = "myarapat/gliner-sports-ner"):
    """
    Load the fine-tuned GLiNER model.

    Args:
        model_path: HuggingFace Hub ID or local checkpoint path.

    Returns:
        Loaded GLiNER model instance.
    """
    from gliner import GLiNER
    print(f"Loading model from {model_path}...")
    model = GLiNER.from_pretrained(model_path, load_tokenizer=True)
    print("Model loaded.")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_extraction(
    text: str,
    entity_types: List[str],
    threshold: float,
    model,
) -> List[Dict]:
    """
    Run GLiNER entity extraction.

    Args:
        text: Raw sports text.
        entity_types: Entity labels to extract.
        threshold: Confidence threshold.
        model: Loaded GLiNER model.

    Returns:
        List of entity dicts with keys: text, label, start, end, score.
    """
    if not text.strip():
        return []
    if not entity_types:
        return []
    return model.predict_entities(text, entity_types, threshold=threshold)


def highlight_entities(text: str, entities: List[Dict]) -> List[Tuple]:
    """
    Convert entity predictions to Gradio HighlightedText format.

    Gradio expects: [(text_segment, label_or_None), ...]

    Args:
        text: Original input text.
        entities: List of entity dicts from run_extraction.

    Returns:
        List of (segment, label) tuples.
    """
    if not entities:
        return [(text, None)]

    # Sort by start offset — essential for correct segmentation
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    segments = []
    cursor = 0

    for ent in sorted_entities:
        start = ent["start"]
        end = ent["end"]

        # Plain text before this entity
        if cursor < start:
            segments.append((text[cursor:start], None))

        # Entity span with label
        segments.append((text[start:end], ent["label"]))
        cursor = end

    # Remaining plain text after last entity
    if cursor < len(text):
        segments.append((text[cursor:], None))

    return segments


# ---------------------------------------------------------------------------
# Gradio event handlers
# ---------------------------------------------------------------------------

def on_extract(text: str, entity_types: List[str], threshold: float, model) -> Tuple:
    """
    Handler for extract button click.
    Returns highlighted text and entity details table.
    """
    if not text.strip():
        return [], pd.DataFrame(columns=["Entity", "Label", "Start", "End", "Score"])

    entities = run_extraction(text, entity_types, threshold, model)
    highlighted = highlight_entities(text, entities)

    # Build details table
    rows = [
        {
            "Entity": e["text"],
            "Label": e["label"],
            "Start": e["start"],
            "End": e["end"],
            "Score": round(e["score"], 4),
        }
        for e in sorted(entities, key=lambda x: x["start"])
    ]
    table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Entity", "Label", "Start", "End", "Score"]
    )

    return highlighted, table


def on_load_benchmark() -> pd.DataFrame:
    """
    Handler for benchmark refresh button.
    Loads benchmark_results.json and returns as DataFrame.
    """
    if not BENCHMARK_PATH.exists():
        return pd.DataFrame({"Error": ["No benchmark results found. Run benchmark.py first."]})

    with open(BENCHMARK_PATH) as f:
        data = json.load(f)

    rows = []
    for r in data["results"]:
        rows.append({
            "System": r["system"],
            "F1": r["f1"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "P50 Latency (ms)": r["p50_latency_ms"],
            "P99 Latency (ms)": r["p99_latency_ms"],
            "Cost / 1K docs": r.get("total_cost_usd", 0.0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------

def extraction_tab(model) -> gr.Tab:
    """Build the entity extraction demo tab."""
    with gr.Tab("Entity Extraction") as tab:
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Sports Article Text",
                    placeholder="Paste a sports article or news snippet here…",
                    value="LeBron James scored 32 points and dished out 11 assists as the Los Angeles Lakers defeated the Golden State Warriors 118-105 at Crypto.com Arena. Head coach Darvin Ham praised James's leadership following the win.",
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

        # Wire button to handler
        extract_btn.click(
            fn=lambda text, types, thresh: on_extract(text, types, thresh, model),
            inputs=[text_input, entity_selector, threshold_slider],
            outputs=[highlighted_output, entity_table],
        )

    return tab


def benchmark_tab() -> gr.Tab:
    """Build the benchmark comparison tab."""
    with gr.Tab("Benchmark Comparison") as tab:
        gr.Markdown("## System Comparison: Base GLiNER vs Fine-Tuned vs GPT-4o-mini")
        gr.Markdown(
            "Fine-tuned GLiNER matches GPT-4o-mini F1 (0.842 vs 0.838) "
            "at **10x lower latency** and **zero per-document cost**."
        )
        benchmark_table = gr.Dataframe(
            headers=["System", "F1", "Precision", "Recall",
                     "P50 Latency (ms)", "P99 Latency (ms)", "Cost / 1K docs"],
            label="Benchmark Results",
        )
        refresh_btn = gr.Button("Load Benchmark Results")

        refresh_btn.click(
            fn=on_load_benchmark,
            inputs=[],
            outputs=[benchmark_table],
        )

    return tab


# ---------------------------------------------------------------------------
# App entrypoint
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble the full Gradio demo application."""
    model = load_model("myarapat/gliner-sports-ner")

    with gr.Blocks(title="GLiNER Sports NER Demo") as demo:
        gr.Markdown("# 🏈 GLiNER Sports NER Demo")
        gr.Markdown(
            "Fine-tuned named entity recognition for sports articles. "
            "Extracts players, teams, stats, injuries, venues, and more. "
            "**Model:** [`myarapat/gliner-sports-ner`](https://huggingface.co/myarapat/gliner-sports-ner) | "
            "**Benchmark:** Fine-tuned GLiNER (0.842 F1) matches GPT-4o-mini (0.838 F1) "
            "at 10x lower latency and zero cost."
        )

        extraction_tab(model)
        benchmark_tab()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )