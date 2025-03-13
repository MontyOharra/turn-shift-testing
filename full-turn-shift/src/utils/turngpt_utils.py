
from argparse import ArgumentParser
from turngpt import TurnGPT
from queue import Queue, Empty
from threading import Event

from typing import Optional

# Initialize the TurnGPT model
def initialize_turngpt_model(
        model_name_or_path: Optional[str] = 'gpt2',
        pretrained: Optional[bool] = True,
        trp_projection_steps: Optional[int] = 1,
        trp_projection_type: Optional[str] = "linear",
        omit_dialog_states: Optional[bool] = False,
        no_train_first_n: Optional[int] = 0,
        learning_rate: Optional[float] = 5e-5,
        weight_loss: Optional[float] = 1.0,
        weight_eos_token: Optional[float] = 1.0,
        weight_regular_token: Optional[float] = 1.0,
        dropout: Optional[float] = 0.1,
) -> TurnGPT:

    # Initialize model
    model = TurnGPT(
        pretrained_model_name_or_path=model_name_or_path,
        pretrained=pretrained,
        trp_projection_steps=trp_projection_steps,
        trp_projection_type=trp_projection_type,
        omit_dialog_states=omit_dialog_states,
        no_train_first_n=no_train_first_n,
        learning_rate=learning_rate,
        weight_loss=weight_loss,
        weight_eos_token=weight_eos_token,
        weight_regular_token=weight_regular_token,
        dropout=dropout
    )
    model.init_tokenizer()
    model.initialize_special_embeddings()
    return model


def calculate_turn_shift_from_live_transcription(
        turn_gpt_model : TurnGPT,
        transcription : Queue,
        turn_shift_probs : Queue,
        stop_event : Event,
    ) -> None:
    """
    Calculate the turn shift probabilities from the transcribed text

    Args:
        transcription (Queue): A queue containing the transcribed text
        turnShiftProbs (Queue): A queue to store the turn shift probabilities
    """
    currTranscription = []
    try:
        while not stop_event.is_set():
            try:
                message = transcription.get(timeout=1)
            except Empty:
                continue  # Wait for new messages if the queue is empty

            # Only add non-empty messages (you can strip whitespace)
            if message.strip():
                currTranscription.append(message)
            
            # Combine the transcription list into one string
            full_turn_list = " ".join(message)
            
            # Check if there is any content before processing
            if not full_turn_list.strip():
                continue  # Skip processing if transcription is empty

            # Process the transcribed text with TurnGPT
            out = turn_gpt_model.string_list_to_trp(message)

            for prob in out["trp_probs"][0]:
                prob *= 10  # Adjust probabilities for visualization

            turn_shift_probs.put((out["trp_probs"][0], out['tokens'][0]))
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping turn shift calculation...")