import os
import re
import numpy as np
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import whisper

# ------------------------
# Convert to WAV
# ------------------------
def convert_to_wav(uploaded_file, save_path="input.wav"):
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    temp_path = "temp_input" + ext

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        if ext == ".mp3":
            audio = AudioSegment.from_file(temp_path, format="mp3")
        elif ext == ".wav":
            audio = AudioSegment.from_file(temp_path, format="wav")
        else:
            raise ValueError("Unsupported file format.")
        audio.export(save_path, format="wav")
    except Exception as e:
        st.error(f"❌ Audio conversion failed: {e}")
        return None

    return save_path

# ------------------------
# Whisper + Translation + gTTS
# ------------------------
def extract_audio_text_and_translate(audio_file, out_file, lang):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file)
        text = result['text']
        st.info(f"Transcribed Text: {text}")
    except Exception as e:
        st.error(f"❌ Whisper failed: {e}")
        return None

    if not text.strip():
        st.warning("⚠️ No valid transcribed text")
        return None 
    lang_model_map = {
        'hi': 'Helsinki-NLP/opus-mt-en-hi',
        'fr': 'Helsinki-NLP/opus-mt-en-fr',
        'de': 'Helsinki-NLP/opus-mt-en-de',
        'es': 'Helsinki-NLP/opus-mt-en-es'
    }

    if lang not in lang_model_map:
        st.error("Unsupported language for translation")
        return None

    try:
        model_name = lang_model_map[lang]
        model = MarianMTModel.from_pretrained(model_name)
        tokenizer = MarianTokenizer.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.info(f"Translated Text: {translated_text}")
        tts = gTTS(text=translated_text, lang=lang)
        tts.save(out_file)
    except Exception as e:
        st.error(f"Translation or TTS failed: {e}")
        return None

# ------------------------
# Voice Cloning with YourTTS
# ------------------------
def voice_clone(in_voice_path,text, lang):
    try:
        wav = preprocess_wav(in_voice_path)
        encoder = VoiceEncoder()
        embed = encoder.embed_utterance(wav)
        np.save("speaker_embedding.npy", embed)

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
        tts.tts_to_file(
            text=text,
            speaker_embedding="speaker_embedding.npy",
            speaker_wav=in_voice_path,
            file_path="output.wav",
            language=lang
        )
    except Exception as e:
        st.error(f"Voice cloning failed: {e}")

# ------------------------
# Streamlit UI
# ------------------------
st.title("🎙️ Multilingual Voice Translator & Cloner")

uploaded_file = st.file_uploader("Upload English Audio (WAV or MP3)", type=["wav", "mp3"])

voice_mode = st.radio("Choose Mode", ["Voice Translation with AI", "Voice Cloning"])

# Language dropdown
your_tts_langs = {"English (en)": "en", "French (fr)": "fr", "Portuguese (pt)": "pt"}
marian_langs = {"Hindi (hi)": "hi", "French (fr)": "fr", "German (de)": "de", "Spanish (es)": "es"}

if voice_mode == "Voice Cloning":
    text = st.text_input("Enter your custom text", placeholder="Type something to clone voice as the input audio")
    lang_choice = st.selectbox("Choose Target Language", list(your_tts_langs.keys()))
    lang_code = your_tts_langs[lang_choice]
else:
    lang_choice = st.selectbox("Choose Target Language", list(marian_langs.keys()))
    lang_code = marian_langs[lang_choice]

# Process button
if uploaded_file and st.button("🚀 Process"):
    wav_path = convert_to_wav(uploaded_file)
    if not wav_path:
        st.stop()

    if voice_mode == "Voice Translation with AI":
        extract_audio_text_and_translate(wav_path, "output.wav", lang=lang_code)
    else:
        voice_clone(wav_path,text,lang=lang_code)

    if os.path.exists("output.wav"):
        audio_bytes = open("output.wav", "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        st.download_button("⬇️ Download Output", audio_bytes, "output.wav")
    else:
        st.error("⚠️ Output audio file not found.")
