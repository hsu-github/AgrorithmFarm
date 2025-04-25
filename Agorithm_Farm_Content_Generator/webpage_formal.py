import streamlit as st
from file_generator_workflow import run_text_creation_workflow
# from function import slide_process
from openai import OpenAI
import tempfile
import os
from MarkItDown import convert_file_to_markdown

import os
import requests
from tempfile import NamedTemporaryFile

# Get API key from environment variables
HARDCODED_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["AWS_ACCESS_KEY_ID"] = st.secrets["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"] = st.secrets["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_DEFAULT_REGION"] = st.secrets["AWS_DEFAULT_REGION"]


def main():
    st.title("Agrorithm Farm")

    #  Automatically assign your hardcoded API key
    if "api_key" not in st.session_state:
        st.session_state.api_key = HARDCODED_API_KEY

    #  Show only model selection if not yet set
    if "llm_model_name" not in st.session_state:
        model_option = st.radio(
            "Choose the LLM model you want to use:",
            [
                "Fine tuned GPT-4o-mini",
                "GPT4o-mini Model"
            ],
            index=0
        )

        if st.button("Continue"):
            if model_option.startswith("Original"):
                st.session_state.llm_model_name = "gpt-4o-mini-2024-07-18"
            else:
                st.session_state.llm_model_name = "ft:gpt-4o-mini-2024-07-18:umd::BJkpoMrt"
            st.rerun()
        return
    run_standard_workflow()

def run_standard_workflow():

    if st.button("🔄 Reset"):
        # ✅ Save the existing API key temporarily
        saved_api_key = st.session_state.get("api_key", "")
        # ❌ Clear all other workflow-related keys
        keys_to_clear = [
            "file_type", "workflow_started", "workflow_step", "user_request",
            "uploaded_files", "uploaded_file_text", "original_post",
            "refined_post", "final_post_history", "refined_history",
            "rag_model", "detail_response", "full_request", "show_related_images",
            "show_back_to_files", "llm_model_name", "file_type_radio", "saved_file_type",
            "final_post_current", "final_version_rich", "slides_json",
            "related_images", "ai_generated_image"
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)
        #  Restore the saved API key to prefill the input
        st.session_state["api_key"] = saved_api_key
        st.rerun()


    if not st.session_state.get("workflow_started", False):
        st.markdown("## ✍️ Enter Your Request")

        # --- Handle content type selection safely ---
        if "file_type_radio" not in st.session_state:
            st.session_state.file_type_radio = None

        # Load the previously selected file type if available
        default_index = 0
        options = ["blog posts", "collateral", "powerpoints", "youtube captions", "grant proposals"]

        saved_type = st.session_state.get("saved_file_type") or st.session_state.get("file_type")
        if saved_type in options:
            default_index = options.index(saved_type)

        file_type = st.radio(
            "Choose the content type (optional):",
            options,
            index=default_index,
            key="file_type_radio"
        )

        # Clear selection using rerun workaround
        if st.session_state.file_type_radio is not None:
            if st.button("🚫 Clear Content Type Selection"):
                # Mark for clearing on next run
                st.session_state.clear_radio = True
                # Also clear saved file type to ensure it doesn't get reselected
                if "saved_file_type" in st.session_state:
                    del st.session_state["saved_file_type"]
                if "file_type" in st.session_state:
                    del st.session_state["file_type"]
                st.rerun()

        # Clear the radio key before re-rendering it
        if st.session_state.get("clear_radio", False):
            del st.session_state["file_type_radio"]
            del st.session_state["clear_radio"]
            st.rerun()

        # User request
        user_request = st.text_area(
            "Describe what you'd like to generate:",
            value=st.session_state.get("user_request", ""),
            key="user_request_input"
        )

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload supporting files (optional):",
            type=["pdf", "docx", "txt", "json", "jsonl", "csv", "xlsx", "pptx", "png", "jpg", "jpeg", "mp4"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        st.session_state.uploaded_files = uploaded_files


        # URL input (optional)
        url_input = st.text_input("Or enter a URL (optional):", key="url_input")
        st.session_state.user_provided_url = url_input.strip()

        # GENERATE
        if st.button("📄 Convert & Start Workflow"):
            file_type = st.session_state.get("file_type_radio", "general")
            st.session_state.file_type = file_type
            st.session_state.saved_file_type = file_type

            combined_file_text = ""
            client = OpenAI(api_key=st.session_state.api_key)

            # Handle uploaded files
            if uploaded_files:
                for i, file in enumerate(uploaded_files, start=1):
                    temp_path = os.path.join(tempfile.gettempdir(), file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.read())
                    markdown_text = convert_file_to_markdown(client, temp_path)
                    combined_file_text += f"\n\n---\n📄 File {i}: {file.name}\n{markdown_text}"

            # Handle URL input by downloading content and converting it
            if url_input.strip():
                response = requests.get(url_input.strip())
                if response.status_code == 200:
                    # Save URL content to temporary file for markdown conversion
                    with NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name

                    markdown_text_from_url = convert_file_to_markdown(client, tmp_file_path)
                    combined_file_text += f"\n\n---\n🔗 URL Content (converted):\n{markdown_text_from_url}"

                    # Clean up temporary file
                    os.remove(tmp_file_path)
                else:
                    st.error(f"Failed to retrieve URL content (status code: {response.status_code}).")

            if not user_request.strip() and not combined_file_text.strip():
                st.error("Please enter a valid request, upload files, or enter a URL.")
            else:
                st.session_state.user_request = user_request.strip()

                if combined_file_text:
                    st.session_state.uploaded_file_text = combined_file_text

                st.session_state.workflow_started = True
                st.rerun()

    # STEP 2: Workflow has started → run the main workflow
    else:
        run_text_creation_workflow(
            user_request=st.session_state.get("user_request", ""),
            api_key=st.session_state.get("api_key", ""),
            file_type=st.session_state.get("file_type", "general"),
            model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18")
        )


if __name__ == "__main__":
    main()
