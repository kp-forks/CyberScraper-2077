"""UI components for the Streamlit app."""

import streamlit as st
import time
import pandas as pd
import io
import re
import csv

# Precompiled regex pattern for extracting data from markdown code blocks
_DATA_BLOCK_PATTERN = re.compile(r'```(csv|excel)\n(.*?)\n```', re.DOTALL)

def display_info_icons():
    if "info_icons_displayed" not in st.session_state:
        st.session_state.info_icons_displayed = True
        st.session_state.info_icons_time = time.time()

    if st.session_state.info_icons_displayed:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; gap: 10px; padding: 20px;">
                <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; max-width: 800px;">
                    <div class="info-box" data-type="enter-url">
                        <h3 style="color: #0066cc;">💻 Enter URL</h3>
                        <p style="color: #000000;">Fetch webpage content for extraction.</p>
                    </div>
                    <div class="info-box" data-type="specify-data">
                        <h3 style="color: #cc6600;">🔍 Specify Data</h3>
                        <p style="color: #000000;">Define what data you want to extract.</p>
                    </div>
                    <div class="info-box" data-type="save-data">
                        <h3 style="color: #006600;">💾 Save Data</h3>
                        <p style="color: #000000;">Save in JSON, CSV, or Excel format.</p>
                    </div>
                    <div class="info-box" data-type="convert-data">
                        <h3 style="color: #cc0000;">🔄 Convert Data</h3>
                        <p style="color: #000000;">Convert between different formats.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if time.time() - st.session_state.info_icons_time > 10 or ("messages" in st.session_state and len(st.session_state.messages) > 0):
            st.session_state.info_icons_displayed = False

def extract_data_from_markdown(text: str | bytes | io.BytesIO) -> str | bytes | io.BytesIO | None:
    """Extract data content from markdown code blocks."""
    if isinstance(text, io.BytesIO):
        return text
    if isinstance(text, bytes):
        text = text.decode('utf-8')

    if match := _DATA_BLOCK_PATTERN.search(text):
        data_type = match.group(1)
        data = match.group(2).strip()
        if data_type == 'excel':
            return io.BytesIO(data.encode())
        return data
    return None

def format_data(data: str | bytes | io.BytesIO, format_type: str) -> pd.DataFrame | None:
    """Format data into a pandas DataFrame."""
    try:
        if isinstance(data, io.BytesIO):
            if format_type == 'excel':
                return pd.read_excel(data, engine='openpyxl')
            data.seek(0)
            return pd.read_csv(data)
        elif isinstance(data, bytes):
            if format_type == 'excel':
                return pd.read_excel(io.BytesIO(data), engine='openpyxl')
            return pd.read_csv(io.BytesIO(data))
        else:
            if format_type == 'csv':
                csv_data = list(csv.reader(io.StringIO(data)))

                if not csv_data:
                    raise ValueError("Empty CSV data")

                max_columns = max(len(row) for row in csv_data)
                padded_data = [row + [''] * (max_columns - len(row)) for row in csv_data]

                headers = padded_data[0]
                unique_headers = []
                for i, header in enumerate(headers):
                    if header == '' or header in unique_headers:
                        unique_headers.append(f'Column_{i+1}')
                    else:
                        unique_headers.append(header)

                df = pd.DataFrame(padded_data[1:], columns=unique_headers)
                # Remove empty columns
                return df.loc[:, (df != '').any(axis=0)]
            elif format_type == 'excel':
                return pd.read_excel(io.BytesIO(data.encode()), engine='openpyxl')
    except Exception as e:
        st.error(f"Error formatting data: {str(e)}")
        return None

def display_message(message):
    content = message["content"]
    if isinstance(content, (str, bytes, io.BytesIO)):
        data = extract_data_from_markdown(content)
        if data is not None:
            if isinstance(data, io.BytesIO) or (isinstance(content, str) and 'excel' in content.lower()):
                df = format_data(data, 'excel')
            else:
                df = format_data(data, 'csv')
            
            if df is not None:
                st.dataframe(df)
            else:
                st.warning("Failed to display data as a table. Showing raw content:")
                st.code(content)
        else:
            st.markdown(content)
    else:
        st.markdown(str(content))