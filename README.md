# Agrorithm Farm

Designed specifically for Capital Area Food Bank (CAFB), Agrorithm Farm is an interactive **Streamlit-based** application that allows users (CAFB staff) to generate high-quality content such as:

- Blog posts  
- PowerPoint presentations  
- YouTube transcripts  
- Grant proposals  
- Marketing collateral  

It supports file uploads, URL ingestion, and manual prompts. The app combines Large Language Models (LLMs), image captioning, Retrieval-Augmented Generation (RAG), and dynamic slide generation into a seamless workflow. 

This project is for the 2025 AI Competition at University of Maryland.

---

## Key Features

### Intelligent Content Generation
- Generate structured blog posts, grant proposals, and collateral using **OpenAI GPT-4o-mini** or a fine-tuned variant.
- Supports **tone and style refinement** with human-in-the-loop interaction.

### Presentation Slide Creation
- Converts prompt + retrieved context into structured **slide JSON**.
- Generates dynamic PowerPoint decks with layouts like:
  - Standard  
  - Pie charts  
  - Progress bars  
  - Comparisons  
- Creates background/cover images via **DALL·E**.

### RAG Model (Retrieval-Augmented Generation)
- Loads and indexes documents (blog posts, slides, collateral, transcripts).
- Retrieves relevant chunks to build high-quality context.
- Utilizes **LangChain vector store** and custom chunking logic.

### Image & Video Processing
- Auto-captions images using the **BLIP model**.
- S3-compatible integration to load **images, videos, and captions**.

### File Support
- Supports: `PDF`, `DOCX`, `TXT`, `CSV`, `PPTX`, `JSONL`, `MP4`, `PNG`, `JPG`, and `URLs`.
- Converts uploaded files using **Microsoft MarkItDown** + OpenAI API.

### Export Options
- Download generated outputs as **PDF** or **PowerPoint**.
- Maintain **version history** of tone/style refinements.

---

## Project Structure

| File | Purpose |
|------|---------|
| `webpage_formal.py` | Main Streamlit app entry point |
| `file_generator_workflow.py` | Core prompt handling and generation workflow |
| `rag_model.py` | RAG model class, context retrieval, and prompt dictionary |
| `human_in_loop.py` | Detail checking, tone/style refinement logic |
| `data_processing.py` | BLIP image captioning, JSONL/text/image/video loading |
| `slides_generator.py` | Slide JSON rendering and PowerPoint generation |
| `image_search.py` | Image prompt matching and DALL·E generation |
| `MarkItDown.py` | File-to-Markdown conversion via Microsoft MarkItDown |

---

## Setup Requirements

- Clone the repository and install all necessary packages by running:  
  ```bash
  pip install -r requirements.txt

- Set up your `st.secrets` for:
  - OpenAI API Key  
  - AWS credentials  
- Streamlit Quill Editor for rich text refinement


Discover and enjoy our app, [AgrorithmFarm](https://agrorithm-farm.streamlit.app/)!
