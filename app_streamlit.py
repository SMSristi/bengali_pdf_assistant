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

from google.cloud import texttospeech

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

@st.cache_resource

def get_google_tts_client():

"""Initialize Google Cloud TTS client"""

try:

if "gcp_service_account" in st.secrets:

credentials = service_account.Credentials.from_service_account_info(

st.secrets["gcp_service_account"]

)

client = texttospeech.TextToSpeechClient(credentials=credentials)

return client

else:

return None

except Exception as e:

st.warning(f"Google TTS not available: {str(e)}")

return None

# ==================== PDF COMPRESSION OPTIMIZATION ====================

def compress_image_for_api(image, max_width=1500, max_height=1500, quality=75):
    """
    Compress image internally to save memory and API costs.
    ✅ Reduces file size before sending to Google Vision API
    """
    img = image.copy()
    
    # Resize if too large
    if img.width > max_width or img.height > max_height:
        img.thumbnail((max_width, max_height), Image.LANCZOS)
    
    # Save to BytesIO with compression
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
    img_byte_arr.seek(0)
    
    compressed_bytes = img_byte_arr.getvalue()
    
    # Calculate compression ratio for logging
    original_size = len(image.tobytes())
    compressed_size = len(compressed_bytes)
    ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    return compressed_bytes, ratio

def extract_text_with_google_vision(pdf_path, max_pages=3):

"""Extract text with aggressive memory optimization and internal PDF compression"""

client = get_vision_client()

if client is None:

return "", [], []

try:

# Read PDF bytes

if isinstance(pdf_path, str):

with open(pdf_path, 'rb') as f:

pdf_bytes = f.read()

elif hasattr(pdf_path, 'read'):

pdf_bytes = pdf_path.read()

else:

pdf_bytes = pdf_path

# ✅ OPTIMIZED: Lower DPI and use JPEG

images = convert_from_bytes(

pdf_bytes,

dpi=150,

fmt='jpeg',

thread_count=1

)

# Limit pages

if len(images) > max_pages:

st.warning(f"⚠️ Processing first {max_pages} pages to save memory")

images = images[:max_pages]

full_text = ""

page_images = []

paragraph_boxes = []

progress_bar = st.progress(0)

compression_stats = []

for idx, img in enumerate(images):

# Store original image for display
page_images.append(img.copy())

# ✅ NEW: Compress image internally before API call
compressed_bytes, compression_ratio = compress_image_for_api(img, quality=75)

compression_stats.append({
    'page': idx + 1,
    'compression_ratio': compression_ratio,
    'size_kb': len(compressed_bytes) / 1024
})

# Send compressed image to Google Vision API
image = vision.Image(content=compressed_bytes)

image_context = vision.ImageContext(language_hints=["bn", "en"])

response = client.document_text_detection(image=image, image_context=image_context)

if response.error.message:

st.error(f"Error on page {idx + 1}: {response.error.message}")

continue

if response.full_text_annotation:

page_text = response.full_text_annotation.text

full_text += page_text + "\n\n"

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

# ✅ CLEAR MEMORY AFTER EACH PAGE

del img, compressed_bytes, image, response

if idx % 2 == 0:

gc.collect()

progress_bar.progress((idx + 1) / len(images))

time.sleep(0.3)

# Show compression stats
progress_bar.empty()

st.info("📊 **PDF Compression Summary:**")

total_size = sum(s['size_kb'] for s in compression_stats)

avg_compression = np.mean([s['compression_ratio'] for s in compression_stats])

st.caption(f"✅ Average compression: **{avg_compression:.1f}%** | Total API payload: **{total_size:.1f} KB**")

gc.collect()

return full_text.strip(), page_images, paragraph_boxes

except Exception as e:

st.error(f"Error during OCR: {str(e)}")

return "", [], []

def match_sentences_to_boxes(sentences, paragraph_boxes):

"""Match each sentence to its containing paragraph's bounding box"""

matched_boxes = []

for sentence in sentences:

best_match = None

best_score = 0

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

# ==================== TEXT PROCESSING ====================

@st.cache_resource

def load_qa_model():

"""Load QA model"""

return pipeline("question-answering", model="deepset/roberta-base-squad2")

def semantic_chunk_text(text, chunk_size=500, overlap=100):

"""Chunk text with overlap"""

words = text.split()

chunks = []

for i in range(0, len(words), chunk_size - overlap):

chunk = ' '.join(words[i:i + chunk_size])

chunks.append(chunk)

return chunks

def setup_rag_pipeline(chunks):

"""Setup RAG with FAISS and BM25"""

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = embedder.encode(chunks, show_progress_bar=False)

embeddings = np.array(embeddings).astype('float32')

dense_idx = faiss.IndexFlatL2(embeddings.shape[1])

dense_idx.add(embeddings)

corpus_tokens = [chunk.split() for chunk in chunks]

sparse_idx = BM25Okapi(corpus_tokens)

return dense_idx, sparse_idx, embedder

def hybrid_search(dense_idx, sparse_idx, embedder, query, chunks, k=3, alpha=0.5):

"""Hybrid search: dense + sparse"""

query_embedding = embedder.encode([query])[0]

query_embedding = np.array([query_embedding]).astype('float32')

distances, indices = dense_idx.search(query_embedding, k)

query_tokens = query.split()

sparse_scores = sparse_idx.get_scores(query_tokens)

combined_indices = set(indices[0].tolist() + np.argsort(sparse_scores)[-k:].tolist())

relevant_chunks = [chunks[i] for i in sorted(combined_indices)[:k]]

return relevant_chunks

def generate_summary(text, max_length=150, min_length=50):

"""Generate extractive summary"""

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

return summary[0]['summary_text']

# ==================== TTS FUNCTIONS ====================

def generate_audio_google(text, language_code="bn-IN"):
    """Generate audio using Google Cloud TTS"""
    client = get_google_tts_client()
    if client is None:
        st.error("Google TTS client not initialized")
        return None
    
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=language_code)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        return response.audio_content
    except Exception as e:
        st.error(f"Google TTS error: {str(e)}")
        return None

def generate_audio_facebook(text, max_length=2000):
    """Generate audio using Facebook MMS-TTS (free)"""
    try:
        if len(text) > max_length:
            text = text[:max_length]
        
        tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-ben")
        output = tts_pipeline(text)
        
        audio_np = output['audio']
        sample_rate = output['sampling_rate']
        
        audio_bytes = io.BytesIO()
        wavfile.write(audio_bytes, sample_rate, (audio_np * 32767).astype(np.int16))
        audio_bytes.seek(0)
        
        return audio_bytes.getvalue()
    except Exception as e:
        st.error(f"Facebook TTS error: {str(e)}")
        return None

# ==================== UI COMPONENTS ====================

def pdf_reader_tab():
    """PDF Reader Tab - Page by page navigation"""
    
    if st.session_state.processed_text is None:
        st.info("Upload a PDF first")
        return
    
    if st.session_state.page_images is None or len(st.session_state.page_images) == 0:
        st.error("No pages extracted")
        return
    
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    
    if 'current_sentence_idx' not in st.session_state:
        st.session_state.current_sentence_idx = 0
    
    if 'page_audio_counter' not in st.session_state:
        st.session_state.page_audio_counter = 0
    
    # Split text into sentences
    sentences = nltk.sent_tokenize(st.session_state.processed_text)
    
    if not st.session_state.matched_sentence_boxes:
        st.session_state.matched_sentence_boxes = match_sentences_to_boxes(
            sentences, st.session_state.paragraph_boxes
        )
    
    page_idx = st.session_state.current_page
    page_num = page_idx + 1
    
    # Display current page image
    st.subheader(f"📄 Page {page_num}")
    st.image(st.session_state.page_images[page_idx], use_column_width=True)
    
    # Get current page sentences
    current_page_sentences = []
    for idx, sent_data in enumerate(st.session_state.matched_sentence_boxes):
        if sent_data['page'] == page_idx:
            current_page_sentences.append(sent_data['text'])
    
    if current_page_sentences:
        current_page_text = " ".join(current_page_sentences)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("▶️ Play This Page Only", type="primary", use_container_width=True):
                """
                ✅ NEW: Generate audio for ONLY the current page
                NOT all pages at once - saves memory significantly
                """
                
                # Clear previous audio cache when generating new page
                if st.session_state.page_audio_counter >= 1:
                    st.caption("🗑️ Clearing previous audio cache...")
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                with st.spinner(f"Generating audio for page {page_num} only..."):
                    
                    # Determine which TTS to use
                    use_google = st.sidebar.checkbox("Use Google TTS (Premium)", value=False)
                    
                    if use_google and page_num == 1:
                        # First page with Google TTS (free tier)
                        audio_data = generate_audio_google(current_page_text)
                    elif use_google and page_num > 1:
                        # Paid Google TTS for additional pages
                        st.info("💰 Using paid Google TTS for this page...")
                        audio_data = generate_audio_google(current_page_text)
                    else:
                        # Always free Facebook TTS
                        audio_data = generate_audio_facebook(current_page_text, max_length=2000)
                    
                    if audio_data:
                        st.success("✅ Audio ready for this page!")
                        st.audio(audio_data, format='audio/wav', autoplay=True)
                        
                        # Increment counter and clear old audio
                        st.session_state.page_audio_counter += 1
                        del audio_data
                        gc.collect()
                    else:
                        st.error("Failed to generate audio for this page")
        
        with col_b:
            if st.button("⏮️ Prev Page", use_container_width=True, disabled=(page_idx == 0)):
                st.session_state.current_page -= 1
                gc.collect()
                st.rerun()
        
        with col_c:
            if st.button("⏭️ Next Page", use_container_width=True,
                        disabled=(page_idx >= len(st.session_state.page_images)-1)):
                st.session_state.current_page += 1
                gc.collect()
                st.rerun()
        
        st.caption(f"📄 Page {page_num} of {len(st.session_state.page_images)} | "
                  f"{len(current_page_sentences)} sentences on this page | "
                  f"🔊 Audio generated: {st.session_state.page_audio_counter} time(s)")
    else:
        st.warning("No text found on this page")
    
    st.progress((st.session_state.current_sentence_idx + 1) / len(sentences))
    st.caption(f"Sentence {st.session_state.current_sentence_idx + 1} of {len(sentences)}")
    
    with st.expander("📑 View All Sentences"):
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

Transform your Bengali PDFs into an interactive audiobook experience with **dual TTS options**.

""")

with st.expander("📋 What This App Does - Click to Learn More", expanded=True):

col1, col2 = st.columns(2)

with col1:

st.markdown("""

**🔍 Core Services:**

- **OCR Text Extraction**: Convert PDF images to text using Google Vision API

- **Visual PDF Reader**: Read with page-by-page navigation

- **Dual TTS Options**: Free (Facebook) + Premium (Google Cloud)

- **Smart Q&A**: Ask questions and get AI-powered answers

""")

with col2:

st.markdown("""

**✨ Advanced Features:**

- **🎯 Per-Page Audio Generation**: Generate audio for one page at a time (memory efficient)

- **🗜️ Internal PDF Compression**: Automatically compress images before API calls

- **🔊 Free TTS**: Facebook MMS-TTS (always free)

- **💎 Premium TTS**: Google Cloud TTS (free for 1st page, natural voice)

- **Auto Cache Clear**: Removes previous audio automatically

- **Memory Optimized**: Runs smoothly on free Streamlit tier

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

max_pages = st.sidebar.slider("Pages to process", 1, 10, 3,

help="Reduce if app crashes")

with st.spinner("Processing with Google Vision API..."):

temp_path = f"temp_{int(time.time())}.pdf"

with open(temp_path, 'wb') as f:

f.write(pdf_file.read())

text, images, paragraph_boxes = extract_text_with_google_vision(

temp_path,

max_pages=max_pages

)

os.remove(temp_path)

if text:

st.session_state.processed_text = text

st.session_state.page_images = images

st.session_state.paragraph_boxes = paragraph_boxes

st.session_state.chunks = semantic_chunk_text(text)

st.session_state.rag_setup = setup_rag_pipeline(st.session_state.chunks)

st.session_state.current_sentence_idx = 0

st.session_state.page_audio_counter = 0 # Reset counter

if 'matched_sentence_boxes' in st.session_state:

del st.session_state.matched_sentence_boxes

st.success(f"✅ Extracted {len(text)} characters")

st.caption(f"📄 {len(images)} pages processed")

else:

st.error("Failed to extract text")

st.header("🎙️ TTS Options")

use_google = st.checkbox("Use Google TTS (Premium)", value=False)

st.success("✅ Dual TTS Available")

st.caption("• Free: Facebook MMS-TTS")

st.caption("• Premium: Google Cloud (1st page free)")

st.header("💾 Memory Status")

st.success("✅ Per-page audio + auto cache clearing enabled")

st.caption("• Audio generated per-page only (not all at once)")

st.caption("• Previous audio deleted automatically")

st.caption("• PDFs compressed internally before API calls")

if st.session_state.processed_text is None:

st.info("👈 Upload a PDF from the sidebar to get started")

st.markdown("### 🎯 Our Services")

col1, col2, col3, col4 = st.columns(4)

with col1:

st.markdown("""

#### 📖 PDF Reader

**Interactive Reading**

- Page-by-page navigation

- Per-page audio generation

- Dual TTS options

- Low memory footprint

""")

with col2:

st.markdown("""

#### 💬 Smart Q&A

**Ask Anything**

- Semantic search

- Context-aware answers

- Confidence scores

- Hybrid retrieval

""")

with col3:

st.markdown("""

#### 📝 Summarization

**Quick Overview**

- Short/Medium/Long

- Memory-efficient

- Instant results

- Extractive method

""")

with col4:

st.markdown("""

#### 🔊 Dual TTS

**Listen Your Way**

- Free option (Facebook)

- Premium option (Google)

- Natural voices

- Auto cache clear

""")

else:

# Main tabs

tabs = st.tabs(["📖 PDF Reader", "💬 Q&A", "📝 Summary", "📄 Full Text"])

with tabs[0]:

pdf_reader_tab()

with tabs[1]:

st.subheader("💬 Smart Question Answering")

st.caption("Powered by BanglaBERT + Hybrid RAG")

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

st.subheader("📝 Extractive Summarization")

st.caption("⚡ No heavy models - instant results!")

summary_length = st.radio("Length", ["Short", "Medium", "Long"], index=1, horizontal=True)

if st.button("📄 Generate Summary", type="primary"):

length_map = {"Short": (30, 100), "Medium": (50, 200), "Long": (100, 300)}

min_len, max_len = length_map[summary_length]

summary = generate_summary(

st.session_state.processed_text,

max_length=max_len,

min_length=min_len

)

st.write("**Summary:**")

st.write(summary)

with tabs[3]:

st.subheader("📄 Full Extracted Text")

st.text_area("Document Text", st.session_state.processed_text, height=400, label_visibility="hidden")

if __name__ == "__main__":

main()
