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
import json
from datetime import datetime
import nltk
from rank_bm25 import BM25Okapi
import re
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ==================== GOOGLE VISION API OCR MODULE ====================
# Set API request limit (for free tier: 1000 requests/month, paid: configurable)
MAX_PAGES_PER_REQUEST = 5  # Limit to avoid quota exhaustion
MONTHLY_REQUEST_LIMIT = 1000  # Adjust based on your quota

@st.cache_resource
def get_vision_client():
    if "gcp_service_account" in st.secrets:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
    else:
        st.error("❌ Google Cloud credentials not found in secrets!")
        return None

def extract_text_with_google_vision(pdf_path):
    """Extract text from PDF using Google Vision API with rate limiting"""
    client = get_vision_client()
    if client is None:
        return "❌ Google Vision API not configured properly"

    try:
        # Read PDF file
        if isinstance(pdf_path, str):
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
        elif hasattr(pdf_path, 'read'):
            pdf_bytes = pdf_path.read()
        else:
            pdf_bytes = pdf_path

        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)

        # Limit number of pages to process
        if len(images) > MAX_PAGES_PER_REQUEST:
            st.warning(f"⚠️ PDF has {len(images)} pages. Processing first {MAX_PAGES_PER_REQUEST} to stay within API limits.")
            images = images[:MAX_PAGES_PER_REQUEST]

        full_text = ""
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, img in enumerate(images):
            status_text.text(f"Processing page {idx + 1}/{len(images)}...")

            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create Vision API image object
            image = vision.Image(content=img_byte_arr)

            # Set language hints for Bengali
            image_context = vision.ImageContext(language_hints=["bn", "en"])

            # Perform OCR with DOCUMENT_TEXT_DETECTION (better for dense text)
            response = client.document_text_detection(
                image=image,
                image_context=image_context
            )

            if response.error.message:
                st.error(f"API Error on page {idx + 1}: {response.error.message}")
                continue

            # Extract text
            if response.full_text_annotation:
                page_text = response.full_text_annotation.text
                full_text += page_text + "\n\n"

            # Update progress
            progress_bar.progress((idx + 1) / len(images))

            # Rate limiting: small delay between requests
            time.sleep(0.5)

        status_text.text("✅ OCR completed!")
        progress_bar.empty()
        status_text.empty()

        return full_text.strip()

    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return ""

# ==================== TTS MODULE ====================
@st.cache_resource
def load_tts_model():
    """Load TTS model (cached)"""
    model = VitsModel.from_pretrained("facebook/mms-tts-ben")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    return model, tokenizer

def generate_audio(text, max_length=1000):
    """Generate audio from text"""
    try:
        model, tokenizer = load_tts_model()
        if len(text) > max_length:
            text = text[:max_length] + "..."
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

# ==================== CHUNKING MODULE ====================
def semantic_chunk_text(text, max_chunk_size=1000, overlap=100):
    """Semantic chunking with Bengali sentence awareness"""
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

# ==================== RAG MODULE ====================
@st.cache_resource
def get_embedder():
    """Get sentence embedder (cached)"""
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def setup_rag_pipeline(chunks):
    """Setup hybrid RAG"""
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dense_index = faiss.IndexFlatL2(embeddings.shape[1])
    dense_index.add(np.array(embeddings).astype('float32'))
    tokenized_chunks = [chunk.split() for chunk in chunks]
    sparse_index = BM25Okapi(tokenized_chunks)
    return dense_index, sparse_index, embedder

def hybrid_search(dense_index, sparse_index, embedder, question, chunks, k=3, alpha=0.5):
    """Hybrid search"""
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

# ==================== QA MODULE ====================
@st.cache_resource
def load_qa_model():
    """Load QA model (cached)"""
    model_name = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline('question-answering', model=model, tokenizer=tokenizer)

# ==================== SUMMARIZATION MODULE ====================
@st.cache_resource
def load_summarization_model():
    """Load summarization model (cached)"""
    try:
        return pipeline(
            "summarization",
            model="csebuetnlp/mT5_multilingual_XLSum",
            tokenizer="csebuetnlp/mT5_multilingual_XLSum"
        )
    except:
        return None

def generate_summary(text, max_length=200, min_length=50):
    """Generate summary"""
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

# ==================== STREAMLIT APP ====================
def main():
    st.set_page_config(
        page_title="Bengali PDF Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 Bengali PDF Assistant - Research Edition")
    st.markdown("""
    **Advanced NLP Pipeline for Bengali Document Analysis**

    *Features: Google Vision OCR • Hybrid RAG • Meta MMS-TTS • BanglaBERT QA • Document Summarization*

    ⚠️ **API Limits:** Processing is limited to {0} pages per PDF. Monthly quota: {1} requests.
    """.format(MAX_PAGES_PER_REQUEST, MONTHLY_REQUEST_LIMIT))

    # Initialize session state
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'rag_setup' not in st.session_state:
        st.session_state.rag_setup = None

    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📄 Upload PDF")
        pdf_file = st.file_uploader("Choose a Bengali PDF file", type=['pdf'])

        if pdf_file is not None and st.button("🔬 Process PDF", type="primary"):
            with st.spinner("Processing PDF with Google Vision API..."):
                # Save uploaded file temporarily
                temp_path = f"temp_{int(time.time())}.pdf"
                with open(temp_path, 'wb') as f:
                    f.write(pdf_file.read())

                # Extract text
                text = extract_text_with_google_vision(temp_path)

                # Clean up temp file
                os.remove(temp_path)

                if text:
                    st.session_state.processed_text = text
                    st.session_state.chunks = semantic_chunk_text(text)
                    st.session_state.rag_setup = setup_rag_pipeline(st.session_state.chunks)
                    st.success(f"✅ Extracted {len(text)} characters from PDF")
                else:
                    st.error("Failed to extract text")

        # API Configuration
        st.header("⚙️ API Configuration")
        st.info("""
        **Setup Instructions:**
        1. Create a Google Cloud project
        2. Enable Vision API
        3. Create service account & download JSON key
        4. Add to Streamlit secrets as `GOOGLE_APPLICATION_CREDENTIALS`
        """)

    # Main content area
    if st.session_state.processed_text is None:
        st.info("👈 Upload a PDF file to get started")
        st.markdown("""
        ### How to use:
        1. Upload a Bengali PDF document
        2. Wait for OCR processing (Google Vision API)
        3. Ask questions, generate summaries, or listen to audio

        ### API Quota Information:
        - **Free Tier:** 1,000 requests/month
        - **Paid Tier:** Custom limits (view in Google Cloud Console)
        - **Rate Limits:** Managed automatically with delays
        """)
    else:
        # Display extracted text preview
        with st.expander("📄 View Extracted Text", expanded=False):
            st.text_area("Full Text", st.session_state.processed_text[:2000] + "...", height=200)

        # Tabs for different features
        tab1, tab2, tab3 = st.tabs(["💬 Question Answering", "📝 Summarization", "📖 Read Aloud"])

        with tab1:
            st.subheader("💬 Ask Questions About Your Document")
            col1, col2 = st.columns([2, 1])

            with col1:
                question = st.text_input("🔍 Your question:", placeholder="Type your question here...")

            with col2:
                num_chunks = st.slider("Context chunks", 1, 5, 3)
                alpha = st.slider("Dense/Sparse balance", 0.0, 1.0, 0.5, 0.1)

            if st.button("💡 Get Answer", type="primary"):
                if question:
                    with st.spinner("Searching and generating answer..."):
                        dense_idx, sparse_idx, embedder = st.session_state.rag_setup
                        relevant_chunks = hybrid_search(
                            dense_idx, sparse_idx, embedder, question,
                            st.session_state.chunks, k=num_chunks, alpha=alpha
                        )
                        context = "\n---\n".join(relevant_chunks)

                        qa_pipeline = load_qa_model()
                        result = qa_pipeline(question=question, context=context)

                        st.success(f"**Answer:** {result['answer']}")
                        st.info(f"**Confidence:** {result['score']:.2%}")

                        with st.expander("📚 View Retrieved Context"):
                            st.text(context)

                        # Generate audio
                        audio_data = generate_audio(result['answer'])
                        if audio_data:
                            st.audio(audio_data, format='audio/wav')
                else:
                    st.warning("Please enter a question")

        with tab2:
            st.subheader("📝 Document Summarization")

            col1, col2 = st.columns([1, 2])
            with col1:
                summary_length = st.radio("Summary Length", ["Short", "Medium", "Long"], index=1)

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

                    # Generate audio
                    audio_data = generate_audio(summary)
                    if audio_data:
                        st.audio(audio_data, format='audio/wav')

        with tab3:
            st.subheader("📖 Listen to Your Document")
            st.info("Generate audio for the first 1000 characters of your document")

            if st.button("🎙️ Generate Audio", type="primary"):
                with st.spinner("Generating audio..."):
                    audio_data = generate_audio(st.session_state.processed_text[:1000])
                    if audio_data:
                        st.success("✅ Audio generated!")
                        st.audio(audio_data, format='audio/wav')

if __name__ == "__main__":
    main()
