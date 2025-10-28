import streamlit as st
from pdf2image import convert_from_bytes
from transformers import pipeline, VitsModel, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import scipy.io.wavfile as wavfile
import io
import time
import gc
from datetime import datetime
import nltk
from rank_bm25 import BM25Okapi
import re
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import os

# NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Memory optimization
torch.set_num_threads(1)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Constants
MAX_PAGES_PER_REQUEST = 5

# ==================== GOOGLE VISION API ====================
@st.cache_resource
def get_vision_client():
    try:
        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return vision.ImageAnnotatorClient(credentials=credentials)
        else:
            st.error("❌ Google Cloud credentials not found!")
            return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def split_into_sentences(text):
    sentences = re.split(r'([।.!?]\s*)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    return [s.strip() for s in result if s.strip()]

def extract_text_with_page_info(pdf_file):
    client = get_vision_client()
    if client is None:
        return "", [], []

    try:
        if hasattr(pdf_file, 'read'):
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)
        else:
            pdf_bytes = pdf_file

        images = convert_from_bytes(pdf_bytes)

        if len(images) > MAX_PAGES_PER_REQUEST:
            st.warning(f"⚠️ Processing first {MAX_PAGES_PER_REQUEST} pages")
            images = images[:MAX_PAGES_PER_REQUEST]

        full_text = ""
        page_images = []
        page_texts = []
        progress_bar = st.progress(0)

        for idx, img in enumerate(images):
            page_images.append(img)

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            image = vision.Image(content=img_byte_arr)
            image_context = vision.ImageContext(language_hints=["bn", "en"])
            response = client.document_text_detection(image=image, image_context=image_context)

            if response.error.message:
                st.error(f"Error on page {idx + 1}")
                page_texts.append("")
                continue

            page_text = ""
            if response.full_text_annotation:
                page_text = response.full_text_annotation.text
                page_texts.append(page_text)
                full_text += page_text + "\n\n"
            else:
                page_texts.append("")

            progress_bar.progress((idx + 1) / len(images))
            time.sleep(0.5)

        progress_bar.empty()
        return full_text.strip(), page_images, page_texts

    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", [], []

def create_sentence_page_map(page_texts):
    sentence_to_page = []
    for page_num, page_text in enumerate(page_texts):
        sentences = split_into_sentences(page_text)
        for sentence in sentences:
            sentence_to_page.append({'text': sentence, 'page': page_num + 1})
    return sentence_to_page

# ==================== TTS ====================
def load_tts_model():
    if 'tts_model' not in st.session_state:
        with st.spinner("Loading TTS..."):
            model = VitsModel.from_pretrained("facebook/mms-tts-ben")
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
            st.session_state.tts_model = (model, tokenizer)
    return st.session_state.tts_model

def generate_audio(text, max_length=300):
    try:
        model, tokenizer = load_tts_model()
        if len(text) > max_length:
            text = text[:max_length]
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        audio_buffer = io.BytesIO()
        wavfile.write(audio_buffer, rate=16000, data=(waveform * 32767).astype(np.int16))
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

# ==================== RAG ====================
def semantic_chunk_text(text, max_chunk_size=1000):
    sentences = re.split(r'[।.!?]\s+', text)
    sentences = [s.strip() + '।' if not s.endswith(('।', '.', '!', '?')) else s.strip()
                 for s in sentences if s.strip()]
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@st.cache_resource
def get_embedder():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def setup_rag_pipeline(chunks):
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dense_index = faiss.IndexFlatL2(embeddings.shape[1])
    dense_index.add(np.array(embeddings).astype('float32'))
    tokenized_chunks = [chunk.split() for chunk in chunks]
    sparse_index = BM25Okapi(tokenized_chunks)
    return dense_index, sparse_index, embedder

def hybrid_search(dense_index, sparse_index, embedder, question, chunks, k=3, alpha=0.5):
    question_embedding = embedder.encode([question])
    dense_distances, dense_indices = dense_index.search(np.array(question_embedding).astype('float32'), k*2)
    tokenized_question = question.split()
    sparse_scores = sparse_index.get_scores(tokenized_question)
    sparse_indices = np.argsort(sparse_scores)[-k*2:][::-1]
    dense_scores = 1 / (1 + dense_distances[0])
    dense_scores = dense_scores / np.sum(dense_scores)
    sparse_scores_norm = sparse_scores[sparse_indices]
    if np.sum(sparse_scores_norm) > 0:
        sparse_scores_norm = sparse_scores_norm / np.sum(sparse_scores_norm)
    combined_scores = {}
    for idx, score in zip(dense_indices[0], dense_scores):
        combined_scores[idx] = combined_scores.get(idx, 0) + alpha * score
    for idx, score in zip(sparse_indices, sparse_scores_norm):
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * score
    top_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunks[idx] for idx, _ in top_indices]

# ==================== QA ====================
def load_qa_model():
    if 'qa_model' not in st.session_state:
        with st.spinner("Loading QA..."):
            model_name = "csebuetnlp/banglabert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            st.session_state.qa_model = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return st.session_state.qa_model

# ==================== SUMMARIZATION ====================
def load_summarization_model():
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading summarizer..."):
            try:
                st.session_state.summarizer = pipeline(
                    "summarization",
                    model="csebuetnlp/mT5_multilingual_XLSum",
                    device_map="auto",
                    load_in_8bit=True,
                    model_kwargs={"torch_dtype": torch.float16}
                )
            except:
                st.session_state.summarizer = None
    return st.session_state.summarizer

def generate_summary(text, max_length=200, min_length=50):
    summarizer = load_summarization_model()
    if summarizer is None:
        return "Model unavailable"
    try:
        if len(text) > 1024:
            text = text[:1024]
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if 'summarizer' in st.session_state:
            del st.session_state.summarizer
            gc.collect()

# ==================== PDF READER (DUAL MODE) ====================
def pdf_reader_tab():
    st.subheader("📖 PDF Reader")

    if 'page_images' not in st.session_state or not st.session_state.page_images:
        st.warning("Please process a PDF first")
        return

    if 'sentence_page_map' not in st.session_state:
        st.session_state.sentence_page_map = create_sentence_page_map(st.session_state.page_texts)

    if 'current_sentence_idx' not in st.session_state:
        st.session_state.current_sentence_idx = 0

    sentence_map = st.session_state.sentence_page_map
    if not sentence_map:
        st.warning("No text found")
        return

    current_item = sentence_map[st.session_state.current_sentence_idx]
    current_sentence = current_item['text']
    current_page = current_item['page']
    page_idx = current_page - 1

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(st.session_state.page_images[page_idx], 
                caption=f"Page {current_page}", 
                use_container_width=True)

    with col2:
        # Mode selection
        reading_mode = st.radio("Mode:", ["🎯 Manual", "▶️ Auto-Play"], horizontal=True)

        st.markdown(f"**Line (Page {current_page}):**")
        st.info(current_sentence)

        # MANUAL MODE
        if reading_mode == "🎯 Manual":
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("⏮️ Prev", disabled=st.session_state.current_sentence_idx == 0):
                    st.session_state.current_sentence_idx -= 1
                    st.rerun()

            with col_b:
                if st.button("🔊 Play"):
                    audio = generate_audio(current_sentence)
                    if audio:
                        st.audio(audio, format='audio/wav', autoplay=True)

            with col_c:
                if st.button("⏭️ Next", disabled=st.session_state.current_sentence_idx >= len(sentence_map)-1):
                    st.session_state.current_sentence_idx += 1
                    st.rerun()

        # AUTO-PLAY MODE
        else:
            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("📄 Read This Page"):
                    page_sentences = [item for item in sentence_map if item['page'] == current_page]

                    for item in page_sentences:
                        idx = sentence_map.index(item)
                        st.session_state.current_sentence_idx = idx

                        audio = generate_audio(item['text'])
                        if audio:
                            st.audio(audio, format='audio/wav', autoplay=True)
                            time.sleep(3)

                    st.success("✅ Page done!")
                    st.rerun()

            with col_b:
                if st.button("📚 Read Entire Book"):
                    start = st.session_state.current_sentence_idx
                    total = len(sentence_map)

                    prog = st.progress(0)

                    for idx in range(start, total):
                        item = sentence_map[idx]
                        st.session_state.current_sentence_idx = idx
                        prog.progress((idx + 1) / total)

                        audio = generate_audio(item['text'])
                        if audio:
                            st.audio(audio, format='audio/wav', autoplay=True)
                            time.sleep(3)

                    prog.empty()
                    st.success("✅ Book done!")
                    st.rerun()

        # Progress
        progress = (st.session_state.current_sentence_idx + 1) / len(sentence_map)
        st.progress(progress)
        st.caption(f"Sentence {st.session_state.current_sentence_idx + 1} / {len(sentence_map)}")

# ==================== MAIN ====================
def main():
    st.set_page_config(page_title="Bengali PDF Assistant", page_icon="🎓", layout="wide")

    st.title("🎓 Bengali PDF Assistant")
    st.markdown("*Google Vision OCR • Dual-Mode Reader • RAG QA • Quantized Summary*")

    # Session state
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'page_images' not in st.session_state:
        st.session_state.page_images = []
    if 'page_texts' not in st.session_state:
        st.session_state.page_texts = []

    # Sidebar
    with st.sidebar:
        st.header("📄 Upload")
        pdf_file = st.file_uploader("PDF", type=['pdf'])

        if pdf_file and st.button("Process", type="primary"):
            with st.spinner("Processing..."):
                text, images, page_texts = extract_text_with_page_info(pdf_file)

                if text:
                    st.session_state.processed_text = text
                    st.session_state.page_images = images
                    st.session_state.page_texts = page_texts
                    st.session_state.chunks = semantic_chunk_text(text)
                    st.session_state.rag_setup = setup_rag_pipeline(st.session_state.chunks)
                    st.session_state.current_sentence_idx = 0
                    st.success(f"✅ {len(text)} chars")
                else:
                    st.error("Failed")

    # Main
    if st.session_state.processed_text is None:
        st.info("👈 Upload PDF")
    else:
        tabs = st.tabs(["📖 Reader", "💬 QA", "📝 Summary"])

        with tabs[0]:
            pdf_reader_tab()

        with tabs[1]:
            st.subheader("💬 Q&A")
            q = st.text_input("Question:")

            if st.button("Answer"):
                if q:
                    dense, sparse, emb = st.session_state.rag_setup
                    chunks = hybrid_search(dense, sparse, emb, q, st.session_state.chunks)
                    qa = load_qa_model()
                    result = qa(question=q, context="\n".join(chunks))
                    st.success(f"**{result['answer']}**")
                    st.info(f"Confidence: {result['score']:.1%}")

        with tabs[2]:
            st.subheader("📝 Summary")
            if st.button("Generate"):
                summary = generate_summary(st.session_state.processed_text[:2000])
                st.write(summary)

if __name__ == "__main__":
    main()
