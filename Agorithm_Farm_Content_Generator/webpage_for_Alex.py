import streamlit as st
from file_generator_workflow import run_text_creation_workflow, reset_entire_system
# from function import slide_process
from openai import OpenAI
import tempfile
import os
from MarkItDown import convert_file_to_markdown
from slides_generator import generate_slide_content, create_presentation, revise_slides, convert_pptx_to_images_mac, convert_pptx_to_images_win
from data_processing import _image_processing

def main():
# ✅ Preload image dictionaries when the app starts
    if "image_data_loaded" not in st.session_state:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(script_dir, ".."))
        RESDIR_PATH = os.path.join(base_dir, "Data", "Raw", "CAFBRain_Dataset")

        with st.spinner("Loading image data...(This may take within a minute)"):
            result_dict = _image_processing(RESDIR_PATH)
            st.session_state.image_data_loaded = True
            st.session_state.collateral_images_dict = result_dict["collateral_images_dict"]
            st.session_state.powerpoints_images_dict = result_dict["powerpoints_images_dict"]
            st.session_state.blog_posts_images_dict = result_dict["blog_posts_images_dict"]

    st.title("Interactive Content Generation Chatbot")

# API Key and Model Selection Input
    if 'api_key' not in st.session_state or 'llm_model_name' not in st.session_state:
        api_key = st.text_input(
            "Enter your OpenAI API key:", 
            type="password",
            value=st.session_state.get('api_key', ''),  # Keeps previous api_key
            key="api_key_input"
        )

        model_option = st.radio(
            "Choose the LLM model you want to use:",
            ["Original (gpt-4o-mini-2024-07-18)", 
            "Fine-tuned (Provide given API key to access the fine-tuned model)"],
            index=0  # Default to original model
        )

        if st.button("Submit API Key"):
            if api_key.strip():
                st.session_state.api_key = api_key
                if model_option.startswith("Original"):
                    st.session_state.llm_model_name = "gpt-4o-mini-2024-07-18"
                else:
                    st.session_state.llm_model_name = "ft:gpt-4o-mini-2024-07-18:personal:BCTJJkOO"
                st.rerun()
            else:
                st.error("Please enter a valid API key.")
        return

    # Select the content type
    if 'file_type' not in st.session_state:
        file_type = st.selectbox(
            "Choose the content type you want to generate:",
            ["blog post", "collateral", "slide", "video script"],
            key="file_type_input"
        )

        # Adjust the spacing with columns to align buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Confirm Content Type"):
                st.session_state.file_type = file_type
                st.rerun()

        with col3:
            if st.button("🔙 Back to Model Selection"):
                if 'llm_model_name' in st.session_state:
                    del st.session_state['llm_model_name']
                st.rerun()
        return
        
#################################################################
    ### Route to appropriate workflow based on file type(if I click "slide", it will go to your slide function):
    if st.session_state.file_type == "slide":
        # Handoff to your coworker's slide generation code
        slide_function(st.session_state.api_key)
    else:
        # Continue with your existing workflow for blog posts and collateral
        run_standard_workflow()
###################################################################

def run_standard_workflow():
    """Handle the standard workflow for blog posts and collateral"""

    if st.button("🔄 Reset"):
        # Completely reset everything including api_key and model choice
        keys_to_clear = [
            "file_type",
            "workflow_started",
            "workflow_step",
            "user_request",
            "uploaded_files",
            "uploaded_file_text",
            "original_post",
            "refined_post",
            "final_post_history",
            "refined_history",
            "rag_model",
            "detail_response",
            "full_request",
            "api_key",
            "llm_model_name",
            "show_related_images",
            "show_back_to_files",
        ]

        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    if st.button("← Back to Content Selection"):
        # Preserve model selection only
        preserved_api_key = st.session_state.get("api_key")
        preserved_model = st.session_state.get("llm_model_name")

        # Clear everything else
        st.session_state.clear()

        # Restore model selection
        if preserved_api_key:
            st.session_state["api_key"] = preserved_api_key
        if preserved_model:
            st.session_state["llm_model_name"] = preserved_model

        # Ensure file_type is reset
        if "file_type" in st.session_state:
            del st.session_state["file_type"]

        st.rerun()

    if not st.session_state.get("workflow_started", False):
        st.markdown(f"### Enter your {st.session_state.file_type} request:")

        user_request = st.text_area(
            "Describe what you'd like to generate:",
            value=st.session_state.get('user_request', ''),
            key='user_request_input'
        )

        uploaded_files = st.file_uploader(
            "Upload files (optional):",
            type=["pdf", "docx", "txt", "json", "jsonl", "csv", "xlsx", "pptx", "png", "jpg", "jpeg", "mp4"],
            accept_multiple_files=True,
            key="file_uploader"
        )

        st.session_state.uploaded_files = uploaded_files

        if st.button(f"📄 Convert & Start {st.session_state.file_type.capitalize()} Workflow"):
            combined_file_text = ""
            if uploaded_files:
                client = OpenAI(api_key=st.session_state.api_key)

                for file in uploaded_files:
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(temp_dir, file.name)

                    with open(temp_path, "wb") as f:
                        f.write(file.read())

                    markdown_text = convert_file_to_markdown(client, temp_path)
                    combined_file_text += f"\n\n---\n\n{markdown_text}"

                st.session_state.uploaded_file_text = combined_file_text

            if not user_request.strip() and not combined_file_text.strip():
                st.error("Please enter a valid request or upload files.")
            else:
                full_request = user_request.strip()
                if combined_file_text:
                    full_request += "\n\n---\n\nUse the following context from uploaded files if helpful:\n" + combined_file_text

                st.session_state.user_request = full_request
                st.session_state.workflow_started = True
                st.rerun()

    else:
        # Workflow already started → run it
        run_text_creation_workflow(
            user_request=st.session_state.get("user_request", ""),
            api_key=st.session_state.get("api_key", ""),
            file_type=st.session_state.get("file_type", "blog post"),
            model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18")
        )


#################################################################
### this is your slide function might look like(defined in slide.py):
### add the api_key parameter to your function and call it here
#################################################################
# Slide function: using generate_slide_content, create_presentation, revise_slides
def slide_function(api_key):
    st.header("Slide Deck Generator")

    if st.button("← Back to Content Selection"):
        if 'file_type' in st.session_state:
            del st.session_state['file_type']
        st.rerun()

    user_query = st.text_input("Enter your presentation request:")
    num_slides = st.number_input("Enter number of slides (excluding title & thank you):", min_value=1, value=10, step=1)
    generate_images = st.checkbox("Generate images with AI", value=False)

    # 🔼 File uploader here
    uploaded_files = st.file_uploader(
        "Upload supporting files (optional):",
        type=["pdf", "docx", "txt", "json", "jsonl", "csv", "xlsx", "pptx", "png", "jpg", "jpeg", "mp4"],
        accept_multiple_files=True
    )

    combined_file_text = ""
    if uploaded_files:
        client = OpenAI(api_key=api_key)
        for file in uploaded_files:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.read())
            markdown_text = convert_file_to_markdown(client, temp_path)
            combined_file_text += f"\n\n---\n\n{markdown_text}"

    # Store slide JSON
    if 'slides_json' not in st.session_state:
        st.session_state.slides_json = None

    if st.button("Generate Slides"):
        if not user_query.strip() and not combined_file_text.strip():
            st.error("Please enter a valid topic or upload files.")
            return

        # 🧠 Merge uploaded content into the prompt
        # full_topic = topic.strip()
        # if combined_file_text:
        #     full_topic += "\n\n---\n\nUse this context from uploaded files:\n" + combined_file_text

        with st.spinner("Generating slide content..."):
            try:
                st.session_state.slides_json = generate_slide_content(
                    user_query=user_query,
                    context=combined_file_text,
                    num_slides=num_slides,
                    api_key=api_key,
                    model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18")  # Fallback if not set
                )
                st.success("Slide content generated successfully!")

                # Create presentation
                slides = st.session_state.slides_json
                client = OpenAI(api_key=api_key)
                temp_filename = slides['title'].replace(" ", "_") + ".pptx"
                create_presentation(st.session_state.slides_json, client=client, filename=temp_filename, generate_image=generate_images)
                
                # Convert and display slides
                st.subheader("Preview Presentation")
                try:
                    # Check operating system and use appropriate conversion function
                    slide_images = convert_pptx_to_images_win(temp_filename) if os.name == 'nt' else convert_pptx_to_images_mac(temp_filename)
                    
                    # Add a scrollable container to view all slides sequentially
                    scroll_container = st.container()
                    with scroll_container:
                        for i, image_path in enumerate(slide_images):
                            st.image(image_path, use_container_width=True)
                            st.markdown("---")
                    
                    # Clean up temporary image files
                    for image_path in slide_images:
                        try:
                            os.remove(image_path)
                        except:
                            pass
                            
                except Exception as e:
                    st.error(f"Error converting slides to images: {e}")
                
                # Add download button
                with open(temp_filename, "rb") as f:
                    ppt_data = f.read()
                st.download_button("Download Presentation", ppt_data, file_name=temp_filename, 
                                 mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                
            except Exception as e:
                st.error(f"Error generating presentation: {e}")
                return

    # Only show revision option if slides have been generated
    if st.session_state.slides_json is not None:
        st.subheader("Revise Slides")
        revision_instruction = st.text_area("Enter revision instructions for the slides:")
        
        if st.button("Apply Revision"):
            with st.spinner("Revising slides..."):
                try:
                    # Update the slides_json in session state with the revised version
                    revised_slides = revise_slides(st.session_state.slides_json, user_query, revision_instruction, combined_file_text, api_key, model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18"))
                    st.session_state.slides_json = revised_slides
                    st.success("Slides revised successfully!")
                    
                    # Create and display revised presentation
                    revised_filename = revised_slides['title'].replace(" ", "_") + "_revised.pptx"
                    client = OpenAI(api_key=api_key)
                    create_presentation(revised_slides, client=client, filename=revised_filename, generate_image=generate_images)
                    
                    # Convert and display revised slides
                    st.subheader("Preview Revised Presentation")
                    
                    # Check operating system and use appropriate conversion function
                    revised_slide_images = convert_pptx_to_images_win(revised_filename) if os.name == 'nt' else convert_pptx_to_images_mac(revised_filename)
                    
                    # Add a scrollable container to view all slides sequentially
                    scroll_container = st.container()
                    with scroll_container:
                        for i, image_path in enumerate(revised_slide_images):
                            st.image(image_path, use_container_width=True)
                            st.markdown("---")  # Add separator between slides
                    
                    # Add download button for revised presentation
                    with open(revised_filename, "rb") as f:
                        revised_ppt_data = f.read()
                    st.download_button("Download Revised Presentation", revised_ppt_data, file_name=revised_filename,
                                     mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                    
                except Exception as e:
                    st.error(f"Error revising slides: {e}")

###################################################################

if __name__ == "__main__":
    main()