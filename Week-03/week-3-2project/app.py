"""Streamlit app: set the temperature and the maximum length, then generate text.

Run:   streamlit run app.py
"""
import streamlit as st

from common import IMAGES_DIR, MODEL_FILE
from generator import generate, load_checkpoint
from utils import repeat_rate

st.set_page_config(page_title="LSTM Text Generator", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner="Loading the model ...")
def get_model():
    return load_checkpoint()


st.title("✍️ LSTM Text Generator")
st.caption("A character-level LSTM that writes one letter at a time, trained on your text corpus.")

if not MODEL_FILE.exists():
    st.error("No trained model found. In the terminal run `python download_corpus.py` and then `python train.py`, "
             "then reload this page.")
    st.stop()

model, vocab, ckpt = get_model()

with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.05,
                            help="Low = safe and repetitive. High = creative but messy.")
    max_length = st.slider("Max length (characters)", 50, 2000, 500, 50)
    top_k = st.slider("Top-k (0 = off)", 0, min(len(vocab), 60), 0,
                      help="Only choose among the k most likely next letters. Cuts out wild typos.")
    fixed = st.checkbox("Repeatable result", value=False, help="Same settings give the same text.")
    random_seed = st.number_input("Random seed", 0, 99999, 42, disabled=not fixed)
    st.divider()
    st.caption(f"Vocabulary: {len(vocab)} characters  \nBest validation loss: {ckpt.get('val_loss', float('nan')):.3f} "
               f"(epoch {ckpt.get('epoch', '?')})")

seed_text = st.text_area("Start the text with (optional)", value="", height=90,
                         placeholder="Type the first words, or leave empty and let the model begin.")

tab_one, tab_compare, tab_about = st.tabs(["Generate", "Compare temperatures", "How it works"])

with tab_one:
    if st.button("Generate", type="primary"):
        with st.spinner("Writing ..."):
            text, removed = generate(model, vocab, seed_text, max_length, temperature, top_k,
                                     int(random_seed) if fixed else None)
        st.session_state["result"] = (text, removed, temperature)
    if "result" in st.session_state:
        text, removed, used_temperature = st.session_state["result"]
        if removed:
            st.warning("These characters are not in the training text and were ignored in your start text: "
                       + " ".join(repr(c) for c in removed))
        st.text_area("Generated text", value=text, height=380)
        left, right = st.columns(2)
        left.metric("Characters", len(text))
        right.metric("Repeated 4-letter chunks", f"{repeat_rate(text) * 100:.0f}%",
                     help="High numbers mean the model is stuck in a loop. Try a higher temperature.")
        st.download_button("Download as .txt", text, file_name="generated.txt")

with tab_compare:
    st.write("Same start text and random seed, three temperatures side by side.")
    if st.button("Compare 0.3 / 0.8 / 1.2"):
        columns = st.columns(3)
        for column, t in zip(columns, (0.3, 0.8, 1.2)):
            text, _ = generate(model, vocab, seed_text, min(max_length, 600), t, top_k, 7)
            column.subheader(f"Temperature {t}")
            column.text(text)

with tab_about:
    st.markdown("""
**What the model does.** It reads text one character at a time. An LSTM keeps a memory of what it has read,
and at every step the model gives a probability for every possible next character. One is picked at random
according to those probabilities and fed back in, so the text grows one letter at a time.

**Temperature** divides the model's scores before they become probabilities:
- below 1: the likely letters get even more likely (safe, but often repeats itself)
- around 1: the probabilities the model learned
- above 1: unlikely letters get a real chance (creative, but more misspellings)

**Max length** is how many new characters are produced. **Top-k** limits each pick to the k most likely letters.
""")
    loss_curve = IMAGES_DIR / "loss_curve.png"
    if loss_curve.exists():
        st.image(str(loss_curve), caption="Training and validation loss")
