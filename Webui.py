import subprocess
import requests
import whisper
import gradio as gr

# -----------------------------------------------------------------------------
# Defaults & Utilities
# -----------------------------------------------------------------------------
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_OLLAMA_MODEL  = "llama3.2:latest"
DEFAULT_SYSTEM_PROMPT = """Você é um assistente técnico experiente em Python, Gradio e IA local.
Sempre responda de forma clara, concisa e forneça exemplos de código quando fizer sentido.""".strip()
DEFAULT_TTS_VOICE     = "Luciana"
DEFAULT_TTS_RATE      = 180  # words per minute

def get_ollama_models() -> list[str]:
    """
    Calls `ollama list` on your machine and returns a list of model names.
    Falls back to [DEFAULT_OLLAMA_MODEL] if it fails.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, check=True
        ).stdout
        lines = [l for l in result.splitlines() if l.strip()]
        # each line: "<model_name>  <other columns…>"
        models = [line.split()[0] for line in lines]
        return models or [DEFAULT_OLLAMA_MODEL]
    except Exception as e:
        print("Could not list Ollama models:", e)
        return [DEFAULT_OLLAMA_MODEL]

# Pre‐seed the dropdown
OLLAMA_MODELS = get_ollama_models()

# -----------------------------------------------------------------------------
# Core pipeline functions
# -----------------------------------------------------------------------------
def transcribe_audio(audio_path: str, model_name: str) -> str:
    model = whisper.load_model(model_name)
    return model.transcribe(audio_path).get("text", "")

def generate_llm_response(user_text: str, llm_model: str, system_prompt: str) -> str:
    prompt = f"{system_prompt}\n\nUsuário: {user_text}\nAssistente:"
    try:
        r = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": llm_model, "prompt": prompt, "stream": False},
            timeout=10
        )
        r.raise_for_status()
        return r.json().get("response", "(sem resposta)")
    except Exception as e:
        return f"❌ Erro ao conectar ao Ollama: {e}"

def synthesize_speech(
    text: str,
    voice: str,
    rate: int,
    aiff_path="response.aiff",
    wav_path="response.wav"
) -> str:
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), "-o", aiff_path, text],
            check=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff_path, wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=True
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
    if not audio_path:
        return "❌ Nenhum áudio enviado.", "", None

    transcript = transcribe_audio(audio_path, whisper_model)
    response   = generate_llm_response(transcript, ollama_model, system_prompt)
    audio_out  = synthesize_speech(response, tts_voice, tts_rate)
    return transcript, response, audio_out

def generate_response_and_audio(
    transcript: str,
    ollama_model: str,
    system_prompt: str,
    tts_voice: str,
    tts_rate: int
):
    if not transcript:
        return "(sem transcrição)", None
    response = generate_llm_response(transcript, ollama_model, system_prompt)
    audio_out = synthesize_speech(response, tts_voice, tts_rate)
    return response, audio_out

# -----------------------------------------------------------------------------
# Callback to “Save Settings” into hidden States
# -----------------------------------------------------------------------------
def save_settings(
    whisper_model, ollama_model, system_prompt, tts_voice, tts_rate
):
    return whisper_model, ollama_model, system_prompt, tts_voice, tts_rate

# -----------------------------------------------------------------------------
# Helper to refresh the Ollama‐models dropdown
# -----------------------------------------------------------------------------
def refresh_ollama_models():
    models = get_ollama_models()
    # Return an “update” to the Dropdown:
    return gr.update(choices=models, value=models[0])

# -----------------------------------------------------------------------------
# Build the Gradio UI
# -----------------------------------------------------------------------------
with gr.Blocks(title="🗣️ STT + LLM + TTS (Refreshable Models)") as demo:
    gr.Markdown("# 🗣️ STT + LLM + TTS\nChoose your model, save settings, then chat!")

    with gr.Tabs():

        # SETTINGS TAB
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("### Configure os parâmetros, depois clique em **Save Settings**")

            whisper_model_dd = gr.Dropdown(
                ["tiny","base","small","medium","large"],
                label="🔊 Whisper Model",
                value=DEFAULT_WHISPER_MODEL
            )
            ollama_model_dd = gr.Dropdown(
                OLLAMA_MODELS,
                label="🤖 Ollama Model",
                value=DEFAULT_OLLAMA_MODEL
            )
            refresh_btn = gr.Button("🔄 Refresh Models")
            system_prompt_ta = gr.Textbox(
                DEFAULT_SYSTEM_PROMPT,
                label="📜 System Prompt",
                lines=4
            )
            tts_voice_dd = gr.Dropdown(
                ["Luciana","Joana","Felipe","Diego","Alex"],
                label="🎙️ macOS Voice",
                value=DEFAULT_TTS_VOICE
            )
            tts_rate_slider = gr.Slider(
                100, 300, step=5, value=DEFAULT_TTS_RATE,
                label="🚀 Speech Rate (wpm)"
            )

            save_btn      = gr.Button("💾 Save Settings")
            save_feedback = gr.Textbox(interactive=False, label="Status")

            # Hidden states for “applied” settings
            whisper_state   = gr.State(DEFAULT_WHISPER_MODEL)
            ollama_state    = gr.State(DEFAULT_OLLAMA_MODEL)
            prompt_state    = gr.State(DEFAULT_SYSTEM_PROMPT)
            tts_voice_state = gr.State(DEFAULT_TTS_VOICE)
            tts_rate_state  = gr.State(DEFAULT_TTS_RATE)

            # Refresh the Ollama dropdown when clicked
            refresh_btn.click(
                fn=refresh_ollama_models,
                inputs=[],
                outputs=ollama_model_dd
            )

            # Save Settings → update the hidden states
            save_btn.click(
                fn=save_settings,
                inputs=[
                    whisper_model_dd,
                    ollama_model_dd,
                    system_prompt_ta,
                    tts_voice_dd,
                    tts_rate_slider
                ],
                outputs=[
                    whisper_state,
                    ollama_state,
                    prompt_state,
                    tts_voice_state,
                    tts_rate_state
                ]
            ).then(
                lambda: "✅ Settings saved! Now go to Interaction.",
                None, save_feedback
            )

        # INTERACTION TAB
        with gr.TabItem("💬 Interaction"):
            gr.Markdown("### Grave ou faça upload de áudio e clique em **Send**")

            audio_input    = gr.Audio(label="🎤 Your Audio", type="filepath")
            send_btn       = gr.Button("▶️ Send")
            transcript_out = gr.Textbox(label="📝 Transcript", interactive=False)
            response_out   = gr.Textbox(label="💡 Assistant Response", interactive=False)
            audio_output   = gr.Audio(label="🔊 Assistant Speech", interactive=False)

            send_btn.click(
                fn=transcribe_audio,
                inputs=[audio_input, whisper_state],
                outputs=[transcript_out]
            ).then(
                fn=generate_response_and_audio,
                inputs=[
                    transcript_out,
                    ollama_state,
                    prompt_state,
                    tts_voice_state,
                    tts_rate_state
                ],
                outputs=[response_out, audio_output]
            )

    demo.launch(
        share=False,
        inbrowser=True,
        server_name="0.0.0.0",
        server_port=7860
    )
