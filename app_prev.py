import gradio as gr
from pdf2image import convert_from_bytes
from transformers import pipeline, VitsModel, AutoTokenizer, AutoModelForQuestionAnswering
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import scipy.io.wavfile as wavfile
import io
import textwrap
import time
import json
from datetime import datetime
import nltk
from rank_bm25 import BM25Okapi
import re
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from PIL import Image
import fitz

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Global variables for caching
ocr_models = None
tts_models = None
qa_model = None
summarization_model = None

# ==================== SURYA OCR MODULE ====================
def load_surya_models():
    """Load Surya OCR models (cached)"""
    global ocr_models
    if ocr_models is None:
        foundation_predictor = FoundationPredictor()
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor(foundation_predictor)
        ocr_models = (det_predictor, rec_predictor)
    return ocr_models

def extract_text_with_surya(pdf_path):
    """Extract text from PDF using Surya OCR"""
    det_predictor, rec_predictor = load_surya_models()
    
    # Read PDF file from path
    if isinstance(pdf_path, str):
        # It's a file path, read it
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
    elif hasattr(pdf_path, 'read'):
        # It's a file object
        pdf_bytes = pdf_path.read()
    else:
        # It's already bytes
        pdf_bytes = pdf_path
    
    images = convert_from_bytes(pdf_bytes)
    
    full_text = ""
    for i, img in enumerate(images):
        det_result = det_predictor([img], ["bn"])
        rec_result = rec_predictor(det_result, [img], ["bn"])
        
        page_text = ""
        for text_line in rec_result[0].text_lines:
            page_text += text_line.text + " "
        
        full_text += page_text + "\n"
    
    return full_text


# ==================== TTS MODULE ====================
def load_tts_model():
    """Load TTS model"""
    global tts_models
    if tts_models is None:
        model = VitsModel.from_pretrained("facebook/mms-tts-ben")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
        tts_models = (model, tokenizer)
    return tts_models

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
def setup_rag_pipeline(chunks):
    """Setup hybrid RAG"""
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
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
def load_qa_model():
    """Load QA model"""
    global qa_model
    if qa_model is None:
        model_name = "csebuetnlp/banglabert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        qa_model = pipeline('question-answering', model=model, tokenizer=tokenizer)
    return qa_model

# ==================== SUMMARIZATION MODULE ====================
def load_summarization_model():
    """Load summarization model"""
    global summarization_model
    if summarization_model is None:
        try:
            summarization_model = pipeline(
                "summarization",
                model="csebuetnlp/mT5_multilingual_XLSum",
                tokenizer="csebuetnlp/mT5_multilingual_XLSum"
            )  # ✅ Add closing parenthesis
        except:
            summarization_model = None
    return summarization_model

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
        )  # ✅ Add closing parenthesis
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"
# ==================== GRADIO INTERFACE FUNCTIONS ====================
def process_pdf(pdf_file, progress=gr.Progress()):
    """Process uploaded PDF"""
    if pdf_file is None:
        return "❌ Please upload a PDF", None, gr.update(visible=False)
    
    try:
        # Get file path from Gradio file object
        pdf_path = pdf_file.name if hasattr(pdf_file, 'name') else str(pdf_file)
        
        # Open PDF to verify it's valid
        doc = fitz.open(pdf_path)
        doc.close()
        
        progress(0, desc="Extracting text from PDF...")
        
        # ✅ FIX: Use pdf_path here!
        text = extract_text_with_surya(pdf_path)
        
        progress(0.5, desc="Setting up RAG pipeline...")
        chunks = semantic_chunk_text(text)
        dense_idx, sparse_idx, embedder = setup_rag_pipeline(chunks)
        
        progress(1.0, desc="Done!")
        
        # Store in state (return as JSON string)
        state = {
            "text": text,
            "chunks": chunks,
            "processed": True
        }
        
        return (
            f"✅ Successfully extracted {len(text)} characters from PDF.\n\n"
            f"📄 Preview:\n{text[:500]}...",
            json.dumps(state),
            gr.update(visible=True)
        )
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}", None, gr.update(visible=False)

def answer_question(question, state_json, num_chunks, alpha):
    """Answer question using RAG"""
    if not state_json:
        return "Please upload and process a PDF first.", None

    try:
        state = json.loads(state_json)
        text = state["text"]
        chunks = state["chunks"]

        # Setup RAG
        dense_idx, sparse_idx, embedder = setup_rag_pipeline(chunks)

        # Search
        relevant_chunks = hybrid_search(
            dense_idx, sparse_idx, embedder, question, chunks,
            k=int(num_chunks), alpha=alpha
        )
        context = "\n---\n".join(relevant_chunks)

        # Get answer
        qa_pipeline = load_qa_model()
        result = qa_pipeline(question=question, context=context)
        answer_text = result['answer']
        confidence = result['score']

        # Generate audio
        audio_data = generate_audio(answer_text)

        response = f"💡 **Answer:** {answer_text}\n\n"
        response += f"📊 **Confidence:** {confidence:.2%}\n\n"
        response += f"📚 **Retrieved Context:**\n{context[:300]}..."

        return response, audio_data
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def generate_summary_fn(state_json, length):
    """Generate document summary"""
    if not state_json:
        return "Please upload and process a PDF first.", None

    try:
        state = json.loads(state_json)
        text = state["text"]

        length_map = {"Short": (30, 100), "Medium": (50, 200), "Long": (100, 300)}
        min_len, max_len = length_map[length]

        summary = generate_summary(text[:2000], max_length=max_len, min_length=min_len)

        # Generate audio
        audio_data = generate_audio(summary)

        return f"📝 **Summary:**\n\n{summary}", audio_data
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def read_aloud(state_json):
    """Generate audio for full document (first 1000 chars)"""
    if not state_json:
        return "Please upload and process a PDF first.", None

    try:
        state = json.loads(state_json)
        text = state["text"]

        # Generate audio for first portion
        audio_data = generate_audio(text[:1000])

        return "🎵 Audio generated for first 1000 characters.", audio_data
    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ==================== GRADIO APP ====================
with gr.Blocks(theme=gr.themes.Soft(), title="Bengali PDF Assistant") as app:
    gr.Markdown("# 🎓 Bengali PDF Assistant - Research Edition")
    gr.Markdown("""
    **Advanced NLP Pipeline for Bengali Document Analysis**  
    *Features: Surya OCR • Hybrid RAG • Meta MMS-TTS • BanglaBERT QA • Document Summarization*
    """)

    state = gr.State(value=None)

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="📄 Upload Bengali PDF", file_types=[".pdf"])
            process_btn = gr.Button("🔬 Process PDF", variant="primary")

            output_text = gr.Textbox(label="Processing Status", lines=10)

    tabs_group = gr.Group(visible=False)

    with tabs_group:
        with gr.Tabs():
            # Tab 1: Q&A
            with gr.Tab("💬 Question Answering"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="🔍 Ask a question about your document",
                            placeholder="Type your question here..."
                        )

                        with gr.Row():
                            num_chunks = gr.Slider(
                                label="Number of context chunks",
                                minimum=1, maximum=5, value=3, step=1
                            )
                            alpha = gr.Slider(
                                label="Dense/Sparse balance",
                                minimum=0.0, maximum=1.0, value=0.5, step=0.1
                            )

                        qa_btn = gr.Button("💡 Get Answer", variant="primary")

                    with gr.Column():
                        qa_output = gr.Textbox(label="Answer", lines=8)
                        qa_audio = gr.Audio(label="🔊 Listen to Answer")

            # Tab 2: Summarization
            with gr.Tab("📝 Summarization"):
                with gr.Row():
                    with gr.Column():
                        summary_length = gr.Radio(
                            ["Short", "Medium", "Long"],
                            label="Summary Length",
                            value="Medium"
                        )
                        summary_btn = gr.Button("📄 Generate Summary", variant="primary")

                    with gr.Column():
                        summary_output = gr.Textbox(label="Summary", lines=8)
                        summary_audio = gr.Audio(label="🔊 Listen to Summary")

            # Tab 3: Read Aloud
            with gr.Tab("📖 Read Aloud"):
                gr.Markdown("Generate audio for the document (first 1000 characters)")
                read_aloud_btn = gr.Button("🎙️ Generate Audio", variant="primary")
                read_aloud_output = gr.Textbox(label="Status", lines=2)
                read_aloud_audio = gr.Audio(label="🔊 Audio")

    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[output_text, state, tabs_group]
    )

    qa_btn.click(
        fn=answer_question,
        inputs=[question_input, state, num_chunks, alpha],
        outputs=[qa_output, qa_audio]
    )

    summary_btn.click(
        fn=generate_summary_fn,
        inputs=[state, summary_length],
        outputs=[summary_output, summary_audio]
    )

    read_aloud_btn.click(
        fn=read_aloud,
        inputs=[state],
        outputs=[read_aloud_output, read_aloud_audio]
    )

    gr.Markdown("""
    ---
    🎓 **Built for academic research & accessibility** | Powered by Surya OCR, BanglaBERT, Meta MMS-TTS
    """)

if __name__ == "__main__":
    app.launch()

