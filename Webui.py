import subprocess
import requests
import whisper
import gradio as gr

# -----------------------------------------------------------------------------
# Default settings
# -----------------------------------------------------------------------------
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_OLLAMA_MODEL  = "llama3.2:latest"
DEFAULT_SYSTEM_PROMPT = """Você é um assistente técnico experiente em Python, Gradio e IA local.
Sempre responda de forma clara, concisa e forneça exemplos de código quando fizer sentido.""".strip()
DEFAULT_TTS_VOICE     = "Luciana"
DEFAULT_TTS_RATE      = 180  # words per minute

# -----------------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------------
def transcribe_audio(audio_path: str, model_name: str) -> str:
    model = whisper.load_model(model_name)
    return model.transcribe(audio_path).get("text", "")

def generate_llm_response(user_text: str, llm_model: str, system_prompt: str) -> str:
    prompt = f"{system_prompt}\n\nUsuário: {user_text}\nAssistente:"
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": llm_model, "prompt": prompt, "stream": False},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("response", "(sem resposta)")
    except Exception as e:
        return f"❌ Erro ao conectar ao Ollama: {e}"

def synthesize_speech(text: str, voice: str, rate: int,
                      aiff_path="response.aiff", wav_path="response.wav") -> str:
    try:
        # macOS say → AIFF
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text],
            check=True
        )
        # ffmpeg AIFF → WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return wav_path
    except subprocess.CalledProcessError as e:
        print("TTS error:", e)
        return None

def process_audio(
    audio_path: str,
    whisper_model: str,
    ollama_model: str,
    system_prompt: str,
    tts_voice: str,
    tts_rate: int
):
    """
    Full pipeline: STT → LLM → TTS, using the *saved* settings.
    """
    if not audio_path:
        return "❌ Nenhum áudio enviado.", "", None

    # 1) Transcribe
    transcript = transcribe_audio(audio_path, whisper_model)

    # 2) LLM response
    response = generate_llm_response(transcript, ollama_model, system_prompt)

    # 3) TTS
    out_audio = synthesize_speech(response, tts_voice, tts_rate)
    return transcript, response, out_audio

# -----------------------------------------------------------------------------
# Settings "apply" callback
# -----------------------------------------------------------------------------
def save_settings(
    w_model, o_model, sys_prompt, tts_voice, tts_rate
):
    """
    Store the selected settings into hidden states.
    """
    return w_model, o_model, sys_prompt, tts_voice, tts_rate

# -----------------------------------------------------------------------------
# Build Gradio UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="🗣️ STT + LLM + TTS (with Save Settings)") as demo:
    gr.Markdown("# 🗣️ STT + LLM + TTS\nConfigure, save, then chime in!")

    # Tab container
    with gr.Tabs():

        # ----- SETTINGS Tab -----
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### Configure os parâmetros e, em seguida, clique em **Save Settings**")

            whisper_model_dd = gr.Dropdown(
                ["tiny","base","small","medium","large"],
                value=DEFAULT_WHISPER_MODEL,
                label="🔊 Whisper Model"
            )
            ollama_model_tb = gr.Textbox(
                value=DEFAULT_OLLAMA_MODEL,
                label="🤖 Ollama Model (ex: llama3.2:latest)"
            )
            system_prompt_ta = gr.Textbox(
                value=DEFAULT_SYSTEM_PROMPT,
                label="📜 System Prompt",
                lines=4
            )
            tts_voice_dd = gr.Dropdown(
                ["Luciana","Joana","Felipe","Diego","Alex"],
                value=DEFAULT_TTS_VOICE,
                label="🎙️ macOS Voice"
            )
            tts_rate_slider = gr.Slider(
                100, 300, step=5, value=DEFAULT_TTS_RATE,
                label="🚀 Speech Rate (wpm)"
            )

            save_btn = gr.Button("💾 Save Settings")
            save_feedback = gr.Textbox(interactive=False, label="Status")

            # Hidden State objects to hold the “active” settings
            whisper_state       = gr.State(DEFAULT_WHISPER_MODEL)
            ollama_state        = gr.State(DEFAULT_OLLAMA_MODEL)
            prompt_state        = gr.State(DEFAULT_SYSTEM_PROMPT)
            tts_voice_state     = gr.State(DEFAULT_TTS_VOICE)
            tts_rate_state      = gr.State(DEFAULT_TTS_RATE)

            # Hook the Save button to update those states
            save_btn.click(
                fn=save_settings,
                inputs=[whisper_model_dd, ollama_model_tb,
                        system_prompt_ta, tts_voice_dd, tts_rate_slider],
                outputs=[whisper_state, ollama_state,
                         prompt_state, tts_voice_state, tts_rate_state]
            ).then(
                # notify user
                lambda: "✅ Settings saved! Switch to Interaction tab.",
                None, save_feedback
            )

        # ----- INTERACTION Tab -----
        with gr.TabItem("💬 Interaction"):
            gr.Markdown("### Grave ou envie áudio e clique em **Send**")

            audio_input   = gr.Audio(label="🎤 Your Audio", type="filepath")
            send_btn      = gr.Button("▶️ Send")
            transcript_out = gr.Textbox(label="📝 Transcript", interactive=False)
            response_out   = gr.Textbox(label="💡 Assistant Response", interactive=False)
            audio_output   = gr.Audio(label="🔊 Assistant Speech", interactive=False)

            # process_audio uses *saved* state values, not the raw widgets
            send_btn.click(
                fn=process_audio,
                inputs=[
                    audio_input,
                    whisper_state,    # ← takes the saved Whisper model
                    ollama_state,     # ← saved Ollama model
                    prompt_state,     # ← saved system prompt
                    tts_voice_state,  # ← saved TTS voice
                    tts_rate_state    # ← saved TTS rate
                ],
                outputs=[transcript_out, response_out, audio_output]
            )

    demo.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )
