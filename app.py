"""
main.py
"""

# Standard library imports
import glob
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Tuple, Optional

# Third-party imports
import gradio as gr
from loguru import logger
from pydantic import BaseModel
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from prompts import SYSTEM_PROMPT
from utils import generate_script, generate_audio

class DialogueItem(BaseModel):
    """A single dialogue item."""

    speaker: Literal["Host (Jane)", "Guest"]
    text: str


class Dialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem]


def generate_podcast(file: str, tone: Optional[str] = None, length: Optional[str] = None) -> Tuple[str, str]:
    """Generate the audio and transcript from the PDF."""
    # Check if the file is a PDF
    if not file.lower().endswith('.pdf'):
        raise gr.Error("Please upload a PDF file.")

    # Read the PDF file and extract text
    try:
        with Path(file).open("rb") as f:
            reader = PdfReader(f)
            text = "\n\n".join([page.extract_text() for page in reader.pages])
    except Exception as e:
        raise gr.Error(f"Error reading the PDF file: {str(e)}")
    
    # Check if the PDF has more than ~150,000 characters
    if len(text) > 100000:
        raise gr.Error("The PDF is too long. Please upload a PDF with fewer than ~100,000 characters.")

    # Modify the system prompt based on the chosen tone and length
    modified_system_prompt = SYSTEM_PROMPT
    if tone:
        modified_system_prompt += f"\n\nTONE: The tone of the podcast should be {tone}."
    if length:
        length_instructions = {
            "Short (1-2 min)": "Keep the podcast brief, around 1-2 minutes long.",
            "Medium (3-5 min)": "Aim for a moderate length, about 3-5 minutes.",
        }
        modified_system_prompt += f"\n\nLENGTH: {length_instructions[length]}"

    # Call the LLM
    llm_output = generate_script(modified_system_prompt, text, Dialogue)
    logger.info(f"Generated dialogue: {llm_output}")

    # Process the dialogue
    audio_segments = []
    transcript = "" # start with an empty transcript
    total_characters = 0

    for line in llm_output.dialogue:
        logger.info(f"Generating audio for {line.speaker}: {line.text}")
        if line.speaker == "Host (Jane)":
            speaker = f"**Jane**: {line.text}"
        else:
            speaker = f"**{llm_output.name_of_guest}**: {line.text}"
        transcript += speaker + "\n\n"
        total_characters += len(line.text)

        # Get audio file path
        audio_file_path = generate_audio(line.text, line.speaker)
        # Read the audio file into an AudioSegment
        audio_segment = AudioSegment.from_file(audio_file_path)
        audio_segments.append(audio_segment)

    # Concatenate all audio segments
    combined_audio = sum(audio_segments)

    # Export the combined audio to a temporary file
    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    combined_audio.export(temporary_file.name, format="mp3")

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    logger.info(f"Generated {total_characters} characters of audio")

    return temporary_file.name, transcript


demo = gr.Interface(
    title="Open NotebookLM",
    description="Convert your PDFs into podcasts with open-source AI models (Llama 3.1 405B and MeloTTS). \n \n Note: Only the text content of the PDF will be processed. Images and tables are not included. The PDF should be no more than 100,000 characters due to the context length of Llama 3.1 405B.",
    fn=generate_podcast,
    inputs=[
        gr.File(
            label="PDF",
            file_types=[".pdf", "file/*"],
        ),
        gr.Radio(
            choices=["Fun", "Formal"],
            label="Tone of the podcast",
            value="casual"
        ),
        gr.Radio(
            choices=["Short (1-2 min)", "Medium (3-5 min)"],
            label="Length of the podcast",
            value="Medium (3-5 min)"
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Markdown(label="Transcript"),
    ],
    allow_flagging="never",
    api_name="generate_podcast",  # Add this line
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3).launch(show_api=True)  # Add queue with concurrency of 5 and show API
