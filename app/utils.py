"""Utility functions for the Streamlit app."""

import time
import streamlit as st
import random

# Loading messages for the cyberpunk theme
_LOADING_MESSAGES = (
    "Ripping data from the net, Silverhand style...",
    "Scraping the web, leaving no trace...",
    "Burning through the data cores...",
    "Jacking in, scraping every byte...",
    "Tearing down the firewall, extracting the goods...",
    "No rules, just raw data extraction...",
    "Slicing through the web's defenses...",
    "No mercy for the web, just pure data...",
    "Scraping the net, one byte at a time...",
    "Crashing through the data barriers, Johnny-style...",
    "Data extraction in progress—neon lights flickering...",
    "Hacking the matrix, one data stream at a time...",
    "Engaging in data warfare, zero tolerance...",
    "Decrypting the web's secrets, no going back...",
    "Plundering the net's underbelly, Cyberpunk style...",
    "Overloading the data circuits, full throttle...",
    "Breach detected—data infiltration in full swing...",
    "Running stealth protocols, data extraction initiated...",
    "Reprogramming the data streams—Neo's got nothing on us...",
    "Surging through the web's dark alleys, data secured...",
    "Unleashing the data chaos—no boundaries...",
    "Breaking through digital fortresses, one byte at a time...",
    "Cracking the net's encryption—data heist in motion...",
    "Navigating the data labyrinth, Cyberpunk flair...",
    "Infiltrating the data vaults—high-tech heist underway...",
)


def get_loading_message() -> str:
    """Get a random cyberpunk-themed loading message."""
    return random.choice(_LOADING_MESSAGES)


def loading_animation(
    process_func,
    *args,
    max_retries: int = 3,
    timeout: float = 60.0,
    **kwargs
):
    """
    Execute a function with loading animation and retry logic.

    Args:
        process_func: The function to execute
        *args: Arguments to pass to the function
        max_retries: Maximum number of retry attempts (default: 3)
        timeout: Timeout in seconds (default: 60)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function, or None on timeout/failure
    """
    loading_placeholder = st.empty()
    result = None
    start_time = time.time()
    retries = 0

    while result is None and retries < max_retries:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            loading_placeholder.error("Request timed out. Please try again.")
            return None

        loading_message = get_loading_message()

        with st.spinner(loading_message):
            try:
                result = process_func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    loading_placeholder.error(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
                # Exponential backoff
                wait_time = min(2 ** retries, 10)
                loading_placeholder.warning(f"Attempt {retries}/{max_retries} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    loading_placeholder.empty()
    if result is not None:
        st.success("Done!")
    return result
