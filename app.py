"""
Streamlit Embedding Visualizer
──────────────────────────────
Upload images and enter text to generate CLIP embeddings and
visualize them on an interactive 2-D scatter plot + similarity heatmap.
"""

import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA

# ── Bootstrap the model once (cached across reruns) ─────────────────────────
# Defer the heavy import so Streamlit can show the page skeleton immediately.

@st.cache_resource(show_spinner="Loading CLIP model …")
def _load_model():
    """Import inference module (which loads the checkpoint) and return helpers."""
    from inference import get_image_embedding, get_text_embedding, checkpoint_type
    if checkpoint_type != "clip_style":
        st.error(
            "The loaded checkpoint is **text-only**. "
            "Train or load a `clip_style` checkpoint to use image embeddings."
        )
    return get_text_embedding, get_image_embedding


get_text_embedding, get_image_embedding = _load_model()


# ── Session state initialisation ────────────────────────────────────────────

if "entries" not in st.session_state:
    st.session_state.entries = []  # list of {"label", "type", "embedding"}


def _add_item(label: str, item_type: str, embedding: torch.Tensor):
    """Append an item to the session."""
    st.session_state.entries.append(
        {
            "label": label,
            "type": item_type,
            "embedding": embedding.detach().cpu().squeeze().numpy(),
        }
    )


# ── Page layout ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Embedding Visualizer",
    page_icon="🔮",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d3d5c;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { color: #aaa; font-size: 0.85rem; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🔮 Embedding Visualizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload images & enter text — see how your CLIP model embeds them</p>',
    unsafe_allow_html=True,
)

# ── Sidebar: inputs ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("➕ Add Items")

    # --- Text input ---
    st.subheader("📝 Text")
    text_input = st.text_input("Enter text", placeholder="a photo of a cat")
    if st.button("Add Text", use_container_width=True, type="primary"):
        if text_input.strip():
            with st.spinner("Encoding text …"):
                emb = get_text_embedding(text_input.strip())
            _add_item(text_input.strip(), "text", emb)
            st.success(f"Added: *{text_input.strip()[:40]}*")
        else:
            st.warning("Enter some text first.")

    st.divider()

    # --- Image input ---
    st.subheader("🖼️ Image")
    uploaded = st.file_uploader(
        "Upload image(s)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if st.button("Add Image(s)", use_container_width=True, type="primary"):
        if uploaded:
            for file in uploaded:
                with tempfile.NamedTemporaryFile(suffix=Path(file.name).suffix, delete=False) as tmp:
                    tmp.write(file.getbuffer())
                    tmp_path = tmp.name
                with st.spinner(f"Encoding {file.name} …"):
                    emb = get_image_embedding(tmp_path)
                _add_item(file.name, "image", emb)
                Path(tmp_path).unlink(missing_ok=True)
            st.success(f"Added {len(uploaded)} image(s)")
        else:
            st.warning("Upload at least one image.")

    st.divider()

    # --- Session controls ---
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.entries = []
        st.rerun()

    st.caption(f"**{len(st.session_state.entries)}** items in session")


# ── Main area: visualisation ────────────────────────────────────────────────

items = st.session_state.entries

# Metric cards
col1, col2, col3 = st.columns(3)
n_text = sum(1 for i in items if i["type"] == "text")
n_image = sum(1 for i in items if i["type"] == "image")

with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{len(items)}</div>'
        f'<div class="metric-label">Total Items</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{n_text}</div>'
        f'<div class="metric-label">Text Items</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-value">{n_image}</div>'
        f'<div class="metric-label">Image Items</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

if len(items) < 2:
    st.info("Add at least **2 items** (text or images) via the sidebar to see the embedding visualisation.", icon="👈")
    st.stop()

# Build matrices
labels = [it["label"] for it in items]
types = [it["type"] for it in items]
embeddings = np.stack([it["embedding"] for it in items])

# ── Scatter plot (PCA 2-D) ──────────────────────────────────────────────────

tab_scatter, tab_heatmap, tab_table = st.tabs([" Scatter Plot", "Similarity Heatmap", "📋 Similarity Table"])

with tab_scatter:
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    color_map = {"text": "#667eea", "image": "#f5a623"}
    colors = [color_map[t] for t in types]
    symbol_map = {"text": "circle", "image": "diamond"}
    symbols = [symbol_map[t] for t in types]

    fig = go.Figure()

    for item_type, marker_symbol, color in [("text", "circle", "#667eea"), ("image", "diamond", "#f5a623")]:
        mask = [i for i, t in enumerate(types) if t == item_type]
        if not mask:
            continue
        fig.add_trace(
            go.Scatter(
                x=coords[mask, 0],
                y=coords[mask, 1],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol=marker_symbol, line=dict(width=1, color="white")),
                text=[labels[i][:25] for i in mask],
                textposition="top center",
                textfont=dict(size=10, color="#ccc"),
                name=item_type.capitalize(),
                hovertext=[labels[i] for i in mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.6)",
        title=dict(text="Embedding Space (PCA 2-D)", font=dict(size=18)),
        xaxis_title=f"PC 1 ({pca.explained_variance_ratio_[0]:.0%} var)",
        yaxis_title=f"PC 2 ({pca.explained_variance_ratio_[1]:.0%} var)",
        height=550,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Similarity heatmap ──────────────────────────────────────────────────────

with tab_heatmap:
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    sim_matrix = F.cosine_similarity(
        emb_tensor.unsqueeze(0), emb_tensor.unsqueeze(1), dim=2
    ).numpy()

    short_labels = [f"{'📝' if t == 'text' else '🖼️'} {l[:20]}" for l, t in zip(labels, types)]

    fig_heat = px.imshow(
        sim_matrix,
        x=short_labels,
        y=short_labels,
        color_continuous_scale="Viridis",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="Cosine Similarity"),
    )
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Pairwise Cosine Similarity", font=dict(size=18)),
        height=max(400, 50 * len(items)),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Similarity table ────────────────────────────────────────────────────────

with tab_table:
    st.markdown("**Top pairwise similarities** (excluding self-pairs)")
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            pairs.append(
                {
                    "Item A": f"{'📝' if types[i] == 'text' else '🖼️'} {labels[i][:35]}",
                    "Item B": f"{'📝' if types[j] == 'text' else '🖼️'} {labels[j][:35]}",
                    "Similarity": f"{sim_matrix[i][j]:.4f}",
                    "Type": f"{types[i]}↔{types[j]}",
                }
            )
    pairs.sort(key=lambda p: float(p["Similarity"]), reverse=True)
    st.dataframe(pairs, use_container_width=True, hide_index=True)
