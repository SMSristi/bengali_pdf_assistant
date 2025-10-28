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
from PIL import Image, ImageDraw
from google.cloud import vision
from google.oauth2 import service_account
import os

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==================== MEMORY OPTIMIZATION ====================
torch.set_num_threads(1)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ==================== GOOGLE VISION API ====================
MAX_PAGES_PER_REQUEST = 5

@st.cache_resource
def get_vision_client():
    """Initialize Google Vision API client"""
    try:
        if "gcp_service_account" in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            client = vision.ImageAnnotatorClient(credentials=credentials)
            return client
        else:
            st.error("❌ Google Cloud credentials not found in secrets!")
            return None
    except Exception as e:
        st.error(f"Failed to initialize Google Vision API: {str(e)}")
        return None

def extract_text_with_google_vision(pdf_path):
    """Extract text with paragraph-level bounding boxes"""
    client = get_vision_client()
    if client is None:
        return "", [], []

    try:
        if isinstance(pdf_path, str):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
        elif hasattr(pdf_path, 'read'):
            pdf_bytes = pdf_path.read()
        else:
            pdf_bytes = pdf_path

        images = convert_from_bytes(pdf_bytes)

        if len(images) > MAX_PAGES_PER_REQUEST:
            st.warning(f"⚠️ Processing first {MAX_PAGES_PER_REQUEST} pages")
            images = images[:MAX_PAGES_PER_REQUEST]

        full_text = ""
        page_images = []
        paragraph_boxes = []
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
                st.error(f"Error on page {idx + 1}: {response.error.message}")
                continue

            if response.full_text_annotation:
                page_text = response.full_text_annotation.text
                full_text += page_text + "\n\n"

                # Extract paragraph boxes
                for page in response.full_text_annotation.pages:
                    for block in page.blocks:
                        for paragraph in block.paragraphs:
                            paragraph_text = ""
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                paragraph_text += word_text + " "

                            vertices = paragraph.bounding_box.vertices
                            paragraph_boxes.append({
                                'text': paragraph_text.strip(),
                                'box': [(v.x, v.y) for v in vertices],
                                'page': idx
                            })

            progress_bar.progress((idx + 1) / len(images))
            time.sleep(0.5)

        progress_bar.empty()
        return full_text.strip(), page_images, paragraph_boxes

    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return "", [], []

# ✅ NEW: Match sentences to paragraph boxes
def match_sentences_to_boxes(sentences, paragraph_boxes):
    """Match each sentence to its containing paragraph's bounding box"""
    matched_boxes = []

    for sentence in sentences:
        best_match = None
        best_score = 0

        # Find paragraph that contains this sentence
        for para in paragraph_boxes:
            if sentence.strip() in para['text']:
                score = len(sentence) / max(len(para['text']), 1)
                if score > best_score:
                    best_score = score
                    best_match = para

        if best_match:
            matched_boxes.append({
                'text': sentence,
                'box': best_match['box'],
                'page': best_match['page']
            })
        else:
            matched_boxes.append({
                'text': sentence,
                'box': None,
                'page': 0
            })

    return matched_boxes

def draw_highlight_on_image(image, bounding_box, color=(255, 255, 0, 100)):
    """Draw semi-transparent highlight on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')

    if bounding_box and len(bounding_box) >= 2:
        x_coords = [point[0] for point in bounding_box]
        y_coords = [point[1] for point in bounding_box]

        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)

        draw.rectangle([left, top, right, bottom], 
                      fill=color, 
                      outline=(255, 200, 0, 255), 
                      width=3)

    return img_copy

# ==================== TTS MODULE ====================
def load_tts_model():
    """Load TTS model only when needed"""
    if 'tts_model' not in st.session_state:
        with st.spinner("Loading TTS model..."):
            model = VitsModel.from_pretrained("facebook/mms-tts-ben")
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
            st.session_state.tts_model = (model, tokenizer)
    return st.session_state.tts_model

def generate_audio(text, max_length=2000):
    """Generate audio from text"""
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

# ==================== CHUNKING ====================
def semantic_chunk_text(text, max_chunk_size=1000):
    """Semantic chunking"""
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

def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'([।.!?]\s*)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    return [s.strip() for s in result if s.strip()]

# ==================== RAG ====================
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
    dense_distances, dense_indices = dense_index.search(
        np.array(question_embedding).astype('float32'), k*2
    )
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
        with st.spinner("Loading Q&A model..."):
            model_name = "csebuetnlp/banglabert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            st.session_state.qa_model = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return st.session_state.qa_model

# ==================== SUMMARIZATION ====================
def load_summarization_model():
    if 'summarizer' not in st.session_state:
        with st.spinner("Loading summarization model (8-bit)..."):
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
        return "Summarization model not available."
    try:
        max_input = 1024
        if len(text) > max_input:
            text = text[:max_input]
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if 'summarizer' in st.session_state:
            del st.session_state.summarizer
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ==================== PDF READER (FIXED) ====================
def pdf_reader_tab():
    """Interactive PDF reader with corrected highlighting"""
    st.subheader("📖 PDF Reader with Text-to-Speech")

    if 'page_images' not in st.session_state or not st.session_state.page_images:
        st.warning("Please process a PDF first in the sidebar")
        return

    page_num = st.selectbox(
        "Select Page",
        range(1, len(st.session_state.page_images) + 1),
        format_func=lambda x: f"Page {x}"
    )

    page_idx = page_num - 1

    col1, col2 = st.columns([1, 1])

    with col1:
        base_image = st.session_state.page_images[page_idx]

        # Get current sentence's bounding box from matched boxes
        current_box = None
        if ('matched_sentence_boxes' in st.session_state and 
            st.session_state.matched_sentence_boxes and
            'current_sentence_idx' in st.session_state and
            st.session_state.current_sentence_idx < len(st.session_state.matched_sentence_boxes)):

            box_data = st.session_state.matched_sentence_boxes[st.session_state.current_sentence_idx]
            if box_data['box'] and box_data['page'] == page_idx:
                current_box = box_data['box']

        # Draw highlight
        if current_box:
            highlighted_image = draw_highlight_on_image(base_image, current_box)
            st.image(highlighted_image, 
                    caption=f"Page {page_num} 🟡 Highlighted", 
                    use_container_width=True)
        else:
            st.image(base_image, 
                    caption=f"Page {page_num}", 
                    use_container_width=True)

    with col2:
        st.markdown("**📄 Extracted Text**")

        if 'processed_text' in st.session_state:
            all_text = st.session_state.processed_text
            sentences = split_into_sentences(all_text)

            # Create matched boxes if not exists
            if ('matched_sentence_boxes' not in st.session_state and 
                'paragraph_boxes' in st.session_state):
                st.session_state.matched_sentence_boxes = match_sentences_to_boxes(
                    sentences, 
                    st.session_state.paragraph_boxes
                )

            if 'current_sentence_idx' not in st.session_state:
                st.session_state.current_sentence_idx = 0

            reading_mode = st.radio(
                "Reading Mode:",
                ["🎯 Manual (Line by line)", "▶️ Auto-Play (Continuous audio)"],
                horizontal=False
            )

            if sentences:
                current_sentence = sentences[st.session_state.current_sentence_idx]

                st.markdown("**Current Line:**")
                st.info(current_sentence)

                # Manual mode
                if reading_mode == "🎯 Manual (Line by line)":
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if st.button("⏮️ Previous", disabled=st.session_state.current_sentence_idx == 0):
                            st.session_state.current_sentence_idx -= 1
                            st.rerun()

                    with col_b:
                        if st.button("🔊 Read Aloud"):
                            audio_data = generate_audio(current_sentence, max_length=500)
                            if audio_data:
                                st.audio(audio_data, format='audio/wav', autoplay=True)

                    with col_c:
                        if st.button("⏭️ Next", disabled=st.session_state.current_sentence_idx >= len(sentences)-1):
                            st.session_state.current_sentence_idx += 1
                            st.rerun()

                # Auto-play mode
                else:
                    st.markdown("**🎵 Continuous Audio**")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        if st.button("📄 Play All", type="primary", use_container_width=True):
                            remaining = sentences[st.session_state.current_sentence_idx:]
                            combined = " ".join(remaining)

                            max_chars = 2000
                            if len(combined) > max_chars:
                                combined = combined[:max_chars]

                            audio_data = generate_audio(combined, max_length=max_chars)
                            if audio_data:
                                st.audio(audio_data, format='audio/wav', autoplay=True)
                                st.session_state.current_sentence_idx = len(sentences) - 1
                                st.balloons()

                    with col_b:
                        if st.button("🔄 Restart", use_container_width=True):
                            st.session_state.current_sentence_idx = 0
                            st.rerun()

                st.progress((st.session_state.current_sentence_idx + 1) / len(sentences))
                st.caption(f"Sentence {st.session_state.current_sentence_idx + 1} of {len(sentences)}")

                with st.expander("📑 View All"):
                    for idx, sent in enumerate(sentences):
                        if idx == st.session_state.current_sentence_idx:
                            st.markdown(f"**➤ {sent}**")
                        else:
                            st.markdown(sent)

# ==================== MAIN APP ====================
def main():
    st.set_page_config(
        page_title="Bengali PDF Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Bengali PDF Assistant")
    st.markdown("""
    **AI-Powered Document Processing & Interactive Reading Platform**

    Transform your Bengali PDFs into an interactive reading experience with advanced AI capabilities.
    """)

    # ✅ EXPLICIT SERVICE DESCRIPTIONS
    with st.expander("📋 What This App Does", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔍 Core Services:**
            - **OCR Text Extraction**: Convert PDF images to editable text using Google Vision API
            - **Visual PDF Reader**: Read PDFs with sentence-by-sentence highlighting
            - **Text-to-Speech**: Listen to Bengali text with natural voice synthesis
            - **Smart Q&A**: Ask questions and get AI-powered answers from your documents
            """)

        with col2:
            st.markdown("""
            **✨ Advanced Features:**
            - **Auto-Play Mode**: Continuous audio playback for hands-free listening
            - **Manual Mode**: Navigate line-by-line with full control
            - **Document Summarization**: AI-generated summaries using mT5 model
            - **Hybrid Search**: Semantic + keyword-based document retrieval
            """)

        st.markdown("""
        **🎯 Perfect For:**
        - 📚 Students reading Bengali textbooks
        - 📄 Researchers analyzing Bengali documents
        - 👂 Audio book enthusiasts
        - 🎓 Language learners
        """)

    # Session state
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'page_images' not in st.session_state:
        st.session_state.page_images = []
    if 'paragraph_boxes' not in st.session_state:
        st.session_state.paragraph_boxes = []
    if 'matched_sentence_boxes' not in st.session_state:
        st.session_state.matched_sentence_boxes = []
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'rag_setup' not in st.session_state:
        st.session_state.rag_setup = None

    # Sidebar
    with st.sidebar:
        st.header("📄 Upload PDF")
        st.caption("Supports Bengali & English text")
        pdf_file = st.file_uploader("Choose a Bengali PDF", type=['pdf'])

        if pdf_file and st.button("🔬 Process PDF", type="primary"):
            with st.spinner("Processing with Google Vision API..."):
                temp_path = f"temp_{int(time.time())}.pdf"
                with open(temp_path, 'wb') as f:
                    f.write(pdf_file.read())

                text, images, paragraph_boxes = extract_text_with_google_vision(temp_path)
                os.remove(temp_path)

                if text:
                    st.session_state.processed_text = text
                    st.session_state.page_images = images
                    st.session_state.paragraph_boxes = paragraph_boxes
                    st.session_state.chunks = semantic_chunk_text(text)
                    st.session_state.rag_setup = setup_rag_pipeline(st.session_state.chunks)
                    st.session_state.current_sentence_idx = 0

                    # Clear old matched boxes
                    if 'matched_sentence_boxes' in st.session_state:
                        del st.session_state.matched_sentence_boxes

                    st.success(f"✅ Extracted {len(text)} characters")
                    st.caption(f"🟡 Found {len(paragraph_boxes)} paragraphs")
                else:
                    st.error("Failed to extract text")

        st.header("💾 Memory Status")
        st.info("✅ Using 8-bit quantized models")
        st.caption("~75% memory reduction")

        st.header("🛠️ Technologies")
        st.caption("""
        • Google Vision API
        • BanglaBERT
        • mT5 (Quantized)
        • FAISS + BM25
        • MMS-TTS
        """)

    # Main content
    if st.session_state.processed_text is None:
        st.info("👈 Upload a PDF to get started")

        # Service cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
            ### 📖 PDF Reader
            Interactive reading with:
            - Visual highlighting
            - Line-by-line navigation
            - Audio playback
            """)

        with col2:
            st.markdown("""
            ### 💬 Smart Q&A
            Ask questions about your document:
            - Semantic search
            - Context-aware answers
            - Confidence scores
            """)

        with col3:
            st.markdown("""
            ### 📝 Summarization
            AI-powered summaries:
            - Short/Medium/Long
            - Memory-efficient
            - Audio support
            """)

        with col4:
            st.markdown("""
            ### 🔊 Text-to-Speech
            Natural voice synthesis:
            - Bengali language
            - Continuous playback
            - Manual control
            """)
    else:
        tabs = st.tabs(["📖 PDF Reader", "💬 Q&A", "📝 Summary", "📄 Full Text"])

        with tabs[0]:
            pdf_reader_tab()

        with tabs[1]:
            st.subheader("💬 Smart Question Answering")
            st.caption("Uses Hybrid RAG (Dense + Sparse retrieval) with BanglaBERT")
            question = st.text_input("Your question:")

            if st.button("💡 Get Answer", type="primary"):
                if question:
                    with st.spinner("Searching..."):
                        dense_idx, sparse_idx, embedder = st.session_state.rag_setup
                        relevant_chunks = hybrid_search(
                            dense_idx, sparse_idx, embedder, question,
                            st.session_state.chunks, k=3, alpha=0.5
                        )
                        context = "\n---\n".join(relevant_chunks)

                        qa_pipeline = load_qa_model()
                        result = qa_pipeline(question=question, context=context)

                        st.success(f"**Answer:** {result['answer']}")
                        st.info(f"**Confidence:** {result['score']:.2%}")

                        with st.expander("📚 Retrieved Context"):
                            st.text(context)
                else:
                    st.warning("Please enter a question")

        with tabs[2]:
            st.subheader("📝 Document Summarization")
            st.caption("Using quantized mT5 model (75% less memory)")

            summary_length = st.radio("Length", ["Short", "Medium", "Long"], index=1, horizontal=True)

            if st.button("📄 Generate Summary", type="primary"):
                with st.spinner("Generating summary..."):
                    length_map = {"Short": (30, 100), "Medium": (50, 200), "Long": (100, 300)}
                    min_len, max_len = length_map[summary_length]

                    summary = generate_summary(
                        st.session_state.processed_text[:2000],
                        max_length=max_len,
                        min_length=min_len
                    )

                    st.write("**Summary:**")
                    st.write(summary)

                    if st.button("🔊 Listen to Summary"):
                        audio_data = generate_audio(summary)
                        if audio_data:
                            st.audio(audio_data, format='audio/wav')

        with tabs[3]:
            st.subheader("📄 Full Extracted Text")
            st.text_area("", st.session_state.processed_text, height=400)

if __name__ == "__main__":
    main()
