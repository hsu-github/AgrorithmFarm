import streamlit as st
from streamlit_quill import st_quill
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from rag_model import RAGModel
from data_processing import _image_processing_filenames
from human_in_loop import detail_checker,build_full_prompt, generate_text, refine_post, final_edit
import os
import json
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
import re
import markdown
from xhtml2pdf import pisa
from slides_generator import generate_slide_content, create_presentation, revise_slides, convert_pptx_to_images_mac, convert_pptx_to_images_win
from openai import OpenAI
from st_files_connection import FilesConnection

aws_path = st.secrets["AWS_path"]

### Function to generate PDF from text
def generate_pdf(markdown_text: str, filename="output.pdf") -> BytesIO:


    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=40, rightMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom styles
    styles.add(ParagraphStyle(name='H1', fontSize=16, leading=20, spaceAfter=12, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='H2', fontSize=13, leading=18, spaceAfter=8))
    styles.add(ParagraphStyle(name='NormalText', fontSize=11, leading=15, spaceAfter=6))

    elements = []

    lines = markdown_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 0.2 * inch))
            continue

        # Header parsing
        if line.startswith("### "):
            elements.append(Paragraph(line[4:], styles["Heading4"]))
        elif line.startswith("## "):
            elements.append(Paragraph(line[3:], styles["H2"]))
        elif line.startswith("# "):
            elements.append(Paragraph(line[2:], styles["H1"]))
        else:
            # Convert bold markdown to reportlab-friendly format
            line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            elements.append(Paragraph(line, styles["NormalText"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


### Function to create a scrollable box for displaying text
def scrollable_markdown(md_text: str, height: int = 400):
    html = markdown.markdown(md_text)
    st.markdown(
        f"""
        <div style="border: 1px solid #ddd; padding: 1em; height: {height}px;
                    overflow-y: auto; background-color: #fafafa; border-radius: 0.5em;">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )

### Function to reset the entire system
def reset_entire_system():
    saved_api_key = st.session_state.get("api_key", None)
    # Clear everything
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Keep API key only
    st.session_state["api_key"] = saved_api_key

def load_rag_documents():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))
    file_path = os.path.join(base_dir, "Data", "Processed", "Text")
    return file_path

def determine_content_type(llm, user_request):
    prompt = f"""
    Given the user's request below, determine whether the content type requested is:
    - "slides" if user clearly requests slides, presentation, ppt, or PowerPoint.
    - "general" if neither clearly applies.

    Respond ONLY with one word: slides, blog, or general.

    User Request: "{user_request.strip()}"
    """

    response = llm.invoke(prompt).content.lower().strip()
    if response not in ["slides", "general"]:
        response = "general"  # Default safety net
    return response

### Function to run the file creation workflow
def run_text_creation_workflow(user_request, api_key, file_type="general", model_name="ft:gpt-4o-mini-2024-07-18:umd::BJkpoMrt", temperature=0):
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini-2024-07-18", temperature=temperature)
    llm_select = ChatOpenAI(openai_api_key=api_key, model_name=model_name, temperature=temperature)

    if "rag_model" not in st.session_state:
        file_path = load_rag_documents()
        
        # Initialize memory
        memory = ConversationBufferMemory(memory_key="chat_history")

        # Create RAG model
        st.session_state.rag_model = RAGModel(
            document_path=file_path,
            llm=llm,
            api_key=api_key,
            do_chunking=True,
            chunk_size=1000,
            chunk_overlap=200,
            top_k=3,
        )

        # Attach memory to rag_model
        st.session_state.rag_model.memory = memory
        st.session_state.rag_model.memory_log = []

    if "workflow_step" not in st.session_state:
        st.session_state.workflow_step = "check_details"
        st.session_state.full_request = user_request

    if st.session_state.workflow_step == "check_details":
        st.write("Checking if more details are needed...")

        # Combine user input (with any additional info) + uploaded file text
        combined = st.session_state.user_request.strip()

        if st.session_state.get("uploaded_file_text", "").strip():
            combined += "\n\n---\n\nConsider the following context from uploaded files:\n"
            combined += st.session_state.uploaded_file_text.strip()

        st.session_state.full_request = combined + f", Type: {file_type}"  # full version passed to the model
        detail_response = detail_checker(llm, st.session_state.rag_model, st.session_state.full_request)

        st.session_state.detail_response = detail_response
        st.session_state.workflow_step = "awaiting_details"
        st.rerun()

    elif st.session_state.workflow_step == "awaiting_details":
        st.write("Model response:")
        # st.write("request and file" ,st.session_state.full_request)
        st.write(st.session_state.detail_response.strip())

        additional_info = st.text_area("Please provide more details if you'd like (optional):", key="details_input")

        col1, spacer, col2 = st.columns([1, 2, 1])

        with col1:
            if st.button("📩 Submit Additional Info"):
                if additional_info.strip():
                    st.session_state.user_request += " " + additional_info.strip()  # only update user_request
                    st.session_state.workflow_step = "check_details"
                    st.rerun()
                else:
                    st.warning("No additional info entered.")

        with col2:
            if st.button("🚀 Generate Anyway"):
                file_type = st.session_state.get("file_type", "general")

                # Build the full prompt first
                st.session_state.full_prompt = build_full_prompt(
                    st.session_state.rag_model,
                    st.session_state.full_request,
                    file_type
                )

                # LLM-based determination if file_type is general
                if file_type == "general":
                    llm = st.session_state.rag_model.llm  # or your preferred llm object
                    determined_type = determine_content_type(llm, st.session_state.full_request)

                    if determined_type == "slides":
                        st.session_state.workflow_step = "handoff_to_slide"
                    else:
                        st.session_state.workflow_step = "generate_post"
                else:
                    # Original logic for explicitly selected file types
                    if file_type == "powerpoints":
                        st.session_state.workflow_step = "handoff_to_slide"
                    else:
                        st.session_state.workflow_step = "generate_post"

                st.session_state.show_back_to_files = False
                st.rerun()


        st.markdown("---")

        # 🧭 Add "Go Back to Add Files" button
        col_back, col_spacer = st.columns([1, 5])
        with col_back:
            if st.button("🔙 Go Back to Upload Files"):
                # Just go back to the file upload screen but keep the user prompt and uploaded file data
                st.session_state.workflow_started = False
                st.session_state.workflow_step = "check_details"
                
                # Keep current file_type and also explicitly save it as saved_file_type
                if "file_type" in st.session_state:
                    st.session_state.saved_file_type = st.session_state.file_type
                
                st.rerun()


    elif st.session_state.workflow_step == "handoff_to_slide":
        user_input = st.session_state.full_prompt.strip()
        slide_function(api_key, user_request=user_input)


    elif st.session_state.workflow_step == "generate_post":
        st.write(f"Generating ...")
        # full_prompt = st.session_state.full_request + f" Type: {file_type}"  # Already contains user request + uploaded file text
        initial_post = generate_text(
            llm_select,
            st.session_state.rag_model,
            full_prompt=st.session_state.full_prompt.strip(),
        )

        st.session_state.original_post = initial_post
        st.session_state.refined_post = None
        st.session_state.final_post_history = []

        st.subheader(f"{(file_type or 'general').capitalize()} Preview")
        scrollable_markdown(st.session_state.original_post)

        # Two buttons on the same row, right-aligned for the second one
        col_left, col_spacer, col_right = st.columns([1, 1, 1])

        with col_left:
            if st.button("🔙 Go Back to Add More Details"):
                st.session_state.workflow_step = "awaiting_details"
                st.rerun()

        # Show tone/style decision below preview
        st.markdown("---")
        st.markdown("### ✨ Do you want to adjust tone and style before final edits?")

        col_yes, col_no = st.columns([1, 1])
        with col_yes:
            if st.button("🎨 Yes, refine tone and style"):
                st.session_state.workflow_step = "tone_style_refinement"
                st.session_state.passed_generate_post = True
                st.rerun()

        with col_no:
            if st.button("🚀 No, continue to final edits"):
                st.session_state.workflow_step = "final_edit"
                st.session_state.final_post_current = st.session_state.original_post
                st.rerun()

    elif st.session_state.workflow_step == "tone_style_refinement":
        st.subheader(f"{file_type.capitalize()} Preview")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📝 Original")
            scrollable_markdown(st.session_state.original_post)
            pdf_buffer = generate_pdf(st.session_state.original_post, "original.pdf")
            st.download_button("📄 Export Original Version", pdf_buffer, "original.pdf", "application/pdf", key="download_original")

        with col2:
            st.markdown("### ✨ Refined (Current)")
            if st.session_state.get("refined_post"):
                scrollable_markdown(st.session_state.refined_post)
                pdf_buffer = generate_pdf(st.session_state.refined_post, "refined.pdf")
                st.download_button("📄 Export Refined Version", pdf_buffer, "refined.pdf", "application/pdf", key="download_refined")
            else:
                st.info("No refined version yet.")

        st.markdown("---")
        st.subheader("🔧 Adjust Tone and Style")
        new_tone = st.text_input("Enter new tone (e.g., compassionate, persuasive):", key="tone_input")
        new_style = st.text_input("Enter new writing style (e.g., narrative, conversational):", key="style_input")

        if "refined_history" not in st.session_state:
            st.session_state.refined_history = []

        col_left, col_spacer, col_right = st.columns([1, 1, 1])

        with col_left:
            if st.button("🎨 Apply Tone and Style"):
                base_post = st.session_state.original_post
                revised_post = refine_post(
                    llm,
                    st.session_state.rag_model,
                    base_post,
                    new_tone,
                    new_style
                )
                
                st.session_state.refined_history.append({
                    "text": revised_post,
                    "tone": new_tone,
                    "style": new_style
                })

                st.session_state.refined_post = revised_post
                st.rerun()

        with col_right:
            if st.button("✅ Continue to Final Review"):
                st.session_state.workflow_step = "final_edit"
                st.rerun()

        # Show Refined Edit History List (add here)
        if st.session_state.refined_history:
            st.markdown("### 🕓 Refined Edit History")
            for i, history_entry in enumerate(reversed(st.session_state.refined_history)):
                label = f"Version {len(st.session_state.refined_history) - i}"
                with st.expander(label):
                    st.markdown(f"**Tone:** {history_entry['tone']}  |  **Style:** {history_entry['style']}")
                    scrollable_markdown(history_entry['text'], height=200)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"🔄 Revert to {label}", key=f"revert_refined_{i}"):
                            st.session_state.refined_post = history_entry["text"]
                            st.rerun()
                    with col2:
                        if st.button(f"🗑 Delete {label}", key=f"delete_refined_{i}"):
                            index = len(st.session_state.refined_history) - 1 - i
                            st.session_state.refined_history.pop(index)
                            if st.session_state.refined_post == history_entry["text"]:
                                st.session_state.refined_post = (
                                    st.session_state.refined_history[-1]["text"]
                                    if st.session_state.refined_history else None
                                )
                            st.rerun()

        if st.button("🔙 Back to Review"):
            st.session_state.workflow_step = "generate_post"
            st.rerun()

# In the "final_edit" workflow step, modify the display order:

    elif st.session_state.workflow_step == "final_edit":
            st.write("### Final Edits")
            edit_request = st.text_area("Describe the changes you'd like to make:")

            col_left, col_spacer, col_right = st.columns([1, 4, 1])

            with col_left:
                if st.button("Apply Final Edits"):
                    base_post = (
                        st.session_state.final_post_history[-1]["text"]
                        if st.session_state.final_post_history
                        else (st.session_state.refined_post or st.session_state.original_post)
                    )
                    final_post = final_edit(llm, st.session_state.rag_model, base_post, edit_request)
                    st.session_state.final_post_history.append({
                        "text": final_post,
                        "change": edit_request
                    })
                    st.session_state.final_post_current = final_post
                    st.rerun()

            with col_right:
                if st.button("Finish"):
                    st.session_state.workflow_step = "done"
                    st.rerun()



            # Insert Final Edit History Here
            if st.session_state.final_post_history:
                st.markdown("### 🕓 Final Edit History")
                for i, history_entry in enumerate(reversed(st.session_state.final_post_history)):
                    label = f"Version {len(st.session_state.final_post_history) - i}"
                    with st.expander(label):
                        st.markdown(f"**Edit request:** {history_entry['change']}")
                        scrollable_markdown(history_entry['text'], height=200)
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"🔄 Revert to {label}", key=f"revert_final_{i}"):
                                index = len(st.session_state.final_post_history) - 1 - i
                                st.session_state.final_post_current = st.session_state.final_post_history[index]["text"]
                                st.rerun()
                        with col2:
                            if st.button(f"🗑 Delete {label}", key=f"delete_final_{i}"):
                                index = len(st.session_state.final_post_history) - 1 - i
                                deleted_entry = st.session_state.final_post_history.pop(index)

                                # If the one we deleted was being shown, switch to the latest remaining one
                                if (
                                    "final_post_current" in st.session_state
                                    and st.session_state.final_post_current == deleted_entry["text"]
                                ):
                                    if st.session_state.final_post_history:
                                        st.session_state.final_post_current = st.session_state.final_post_history[-1]["text"]
                                    else:
                                        st.session_state.final_post_current = None

                                st.rerun()


            # Show original and most recent (or reverted) final edit side-by-side
            if st.session_state.final_post_history:
                if "final_post_current" in st.session_state and st.session_state.final_post_current:
                    latest_final_edit = {"text": st.session_state.final_post_current}
                else:
                    latest_final_edit = st.session_state.final_post_history[-1]


                base_left = st.session_state.refined_post or st.session_state.original_post
                base_label = "✨ Refined Version" if st.session_state.refined_post else "📝 Original Version"

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"### {base_label}")
                    scrollable_markdown(base_left)
                    pdf_buffer = generate_pdf(base_left, "base_left.pdf")
                    st.download_button(
                        f"📄 Export {base_label}",
                        pdf_buffer,
                        file_name=base_label.lower().replace(" ", "_") + ".pdf",
                        mime="application/pdf",
                        key="finaledit_download_left"
                    )

                with col2:
                    st.markdown("### 🛠️ Final Edit")
                    scrollable_markdown(latest_final_edit["text"])
                    pdf_buffer = generate_pdf(latest_final_edit["text"], "final_edit_latest.pdf")
                    st.download_button(
                        "📄 Export Final Edit",
                        pdf_buffer,
                        "final_edit_latest.pdf",
                        "application/pdf",
                        key="finaledit_download_latest"
                    )

            if st.button("🔙 Back to Tone & Style Editing"):
                st.session_state.workflow_step = "tone_style_refinement"
                st.rerun()

    elif st.session_state.workflow_step == "done":
        if "related_images" not in st.session_state:
            st.session_state.related_images = None
        
        # Add this new session state variable to store AI generated image data
        if "ai_generated_image" not in st.session_state:
            st.session_state.ai_generated_image = None

        st.success(f"{file_type.capitalize()} generation completed!")

        # Get final version as fallback
        if st.session_state.final_post_history:
            final_version = st.session_state.final_post_history[-1]["text"]
        elif st.session_state.refined_post:
            final_version = st.session_state.refined_post
        else:
            final_version = st.session_state.original_post

        # If no previous rich edit, convert markdown to HTML once
        if "final_version_rich" not in st.session_state:
            st.session_state.final_version_rich = markdown.markdown(final_version)

        st.markdown("### ✏️ Final Editor (Rich Format)")

        # Rich editor
        quill_html = st_quill(value=st.session_state.final_version_rich, html=True, key="quill_editor")

        if st.button("✅ Apply Changes"):
            st.session_state["final_version_rich"] = quill_html
            st.success("✅ Changes applied.")

        def generate_pdf_from_html(html_str):
            buffer = BytesIO()
            
            # First, normalize all the HTML content
            cleaned_html = re.sub(r'<p[^>]*>', '<p>', html_str)
            cleaned_html = re.sub(r'<p>\s*</p>', '', cleaned_html)
            cleaned_html = re.sub(r'<br\s*/?>', '', cleaned_html)
            
            # Control line height more strictly in the CSS
            styled_html = f"""
            <html>
            <head>
                <style>
                    @page {{
                        size: A4;
                        margin: 1in;
                    }}
                    body {{
                        font-family: 'Times-Roman';
                        font-size: 12pt;
                        line-height: 1.2;  /* Stricter line height control */
                    }}
                    p {{
                        margin-top: 0;
                        margin-bottom: 0.3em;  /* Reduced bottom margin */
                        line-height: 1.2;  /* Enforce line height at paragraph level */
                    }}
                    h1, h2, h3 {{
                        margin-top: 0.8em;
                        margin-bottom: 0.3em;
                        line-height: 1.2;  /* Consistent line height */
                    }}
                    /* Override any inline styles that might affect spacing */
                    span, strong, em, b, i {{
                        line-height: inherit !important;
                    }}
                </style>
            </head>
            <body>
                {cleaned_html}
            </body>
            </html>
            """
            
            pisa.CreatePDF(src=styled_html, dest=buffer)
            buffer.seek(0)
            return buffer

        pdf_buffer = generate_pdf_from_html(st.session_state["final_version_rich"])

        st.download_button(
            label="📄 Export Final Version to PDF",
            data=pdf_buffer,
            file_name=f"{file_type.replace(' ', '_')}_final.pdf",
            mime="application/pdf",
            key="final_rich_export"
        )

        if st.button("🔙 Back to Final Edit"):
            st.session_state.workflow_step = "final_edit"
            st.rerun()

            # Existing code for image generation...
        if file_type != "video script":
            st.markdown("---")
            st.subheader("🔍 Generate Images")

            # Check if we already have generated related images
            if st.session_state.related_images is not None:
                # Display previously generated related images
                st.markdown("### 🖼 Matched Historical Images")
                cols = st.columns(4)
                for i, img in enumerate(st.session_state.related_images):
                    if img is not None:
                        with cols[i % 4]:
                            # Create a connection to S3
                            conn = st.connection('s3', type=FilesConnection)
                            # Get the image directly from AWS S3
                            image_path = os.path.join(aws_path, 'images', img) 
                            with conn.open(image_path, 'rb') as s3_file:
                                image_bytes = s3_file.read()

                                st.image(image_bytes)
                
                # Add option to regenerate related images
                if st.button("🔄 Regenerate Related Images"):
                    st.session_state.related_images = None
                    st.rerun()
            else:
                # Only show the generate button if we don't have images yet
                if st.button("🖼 Generate Related Images From History Documents"):
                    if "image_data_loaded" not in st.session_state:
                        RESDIR_PATH = st.secrets["AWS_path"]
                
                        with st.spinner("Loading image data...(This may take within a minute)"):
                            result_dict = _image_processing_filenames(RESDIR_PATH)
                            print(result_dict)
                            st.session_state.image_data_loaded = True
                            st.session_state.collateral_images_dict = result_dict["collateral_image_filenames_dict"]
                            st.session_state.powerpoints_images_dict = result_dict["powerpoints_image_filenames_dict"]
                            st.session_state.blog_posts_images_dict = result_dict["blog_posts_image_filenames_dict"]
                
                    with st.spinner("Processing images from history documents..."):
                        rag = st.session_state.rag_model
                        matched_images = rag.collect_images_from_dicts(
                            st.session_state.collateral_images_dict,
                            st.session_state.powerpoints_images_dict,
                            st.session_state.blog_posts_images_dict
                        )
                        st.session_state.related_images = matched_images  # Save result
                        st.rerun()



def slide_function(api_key, user_request):
    st.header("Slide Deck Generator")
    if st.button("← Back to Content Selection"):
        if 'file_type' in st.session_state:
            del st.session_state['file_type']
        st.rerun()

    num_slides = st.number_input("Enter number of slides (excluding title & thank you):", min_value=1, value=10, step=1)
    generate_images = st.checkbox("Generate images with AI", value=False)
    print(num_slides)

    # Store slide JSON
    if 'slides_json' not in st.session_state:
        st.session_state.slides_json = None

    if st.button("Generate Slides"):
        with st.spinner("Crafting your slide deck content..."):
                st.session_state.slides_json = generate_slide_content(
                    final_prompt=user_request,  # topic is now user_query
                    num_slides=num_slides,
                    api_key=api_key,
                    model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18")
                )

                # Create presentation
                slides = st.session_state.slides_json
                client = OpenAI(api_key=api_key)
                temp_filename = slides['title'].replace(" ", "_") + ".pptx"
                create_presentation(st.session_state.slides_json, st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18") , client=client, filename=temp_filename, generate_image=generate_images)
                

        # Convert and display slides
        st.subheader("Preview Presentation")

        # Check operating system and use appropriate conversion function
        with st.spinner("Preparing slide preview images..."):
            slide_images = convert_pptx_to_images_win(temp_filename) if os.name == 'nt' else convert_pptx_to_images_mac(temp_filename)
        
        st.success("Slide content generated successfully!")
        
        # Add a scrollable container to view all slides sequentially
        scroll_container = st.container()
        with scroll_container:
            for i, image_path in enumerate(slide_images):
                st.image(image_path)
                st.markdown("---")
        
        # Clean up temporary image files
        for image_path in slide_images:
            try:
                os.remove(image_path)
            except:
                pass


        
        # Add download button
        with open(temp_filename, "rb") as f:
            ppt_data = f.read()
        st.download_button("Download Presentation", ppt_data, file_name=temp_filename, 
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")


    # Only show revision option if slides have been generated
    if st.session_state.slides_json is not None:
        st.subheader("Revise Slides")
        revision_instruction = st.text_area("Enter revision instructions for the slides:")
        
        if st.button("Apply Revision"):
            with st.spinner("Revising slides..."):
                try:
                    # Update the slides_json in session state with the revised version
                    revised_slides = revise_slides(st.session_state.slides_json, revision_instruction, api_key, model_name=st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18"))
                    st.session_state.slides_json = revised_slides
                    st.success("Slides revised successfully!")
                    
                    # Create and display revised presentation
                    revised_filename = revised_slides['title'].replace(" ", "_") + "_revised.pptx"
                    client = OpenAI(api_key=api_key)
                    print(revise_slides)
                    create_presentation(revised_slides, st.session_state.get("llm_model_name", "gpt-4o-mini-2024-07-18"), client=client, filename=revised_filename, generate_image=generate_images)
                    
                    # Convert and display revised slides
                    st.subheader("Preview Revised Presentation")
                    
                    # Check operating system and use appropriate conversion function
                    revised_slide_images = convert_pptx_to_images_win(revised_filename) if os.name == 'nt' else convert_pptx_to_images_mac(revised_filename)
                    
                    # Add a scrollable container to view all slides sequentially
                    scroll_container = st.container()
                    with scroll_container:
                        for i, image_path in enumerate(revised_slide_images):
                            st.image(image_path)
                            st.markdown("---")  # Add separator between slides
                    
                    # Add download button for revised presentation
                    with open(revised_filename, "rb") as f:
                        revised_ppt_data = f.read()
                    st.download_button("Download Revised Presentation", revised_ppt_data, file_name=revised_filename,
                                     mime="application/vnd.openxmlformats-officedocument.presentationml.presentation")
                    
                except Exception as e:
                    st.error(f"Error revising slides: {e}")
