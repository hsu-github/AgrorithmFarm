# Agrorithm Farm

This project is the winner of **2025 AI Competition at University of Maryland**,[Demo Video](https://drive.google.com/file/d/166Rw3aeZSj4ETnue48w3gdzFX6KjTPtu/view?usp=sharing)

Designed specifically for Capital Area Food Bank (CAFB), Agrorithm Farm is an interactive **Streamlit-based** application that allows users (CAFB staff) to generate CAFB-specific content such as:

- Blog posts  
- PowerPoint presentations  
- YouTube transcripts  
- Grant proposals  
- Marketing collateral  

Agrorithm Farm allows users to make any requests through prompts abd supports file uploads and/or URL ingestion. The app combines Large Language Models (LLMs), image captioning, Retrieval-Augmented Generation (RAG), and dynamic slide generation into a seamless workflow. 

---

## Key Features

### Intelligent Content Generation
- Generates structured blog posts, grant proposals, and collateral using a fine-tuned **OpenAI GPT-4o-mini** model.
- Allows users to uploaed a template for these documents.
- Supports **tone and style refinement** with human-in-the-loop interaction.

### Presentation Slide Creation
- Converts prompt + retrieved context into structured **slide JSON**.
- Generates dynamic PowerPoint decks with slides like:
  - Standard  
  - Pie charts  
  - Progress bars  
  - Comparisons  
- Creates background/cover images via **DALL·E**.

### RAG Model (Retrieval-Augmented Generation)
- Loads and indexes documents (blog posts, slides, collateral, transcripts).
- Retrieves relevant chunks to build context.
- Utilizes **LangChain vector store** and custom chunking logic.

### Image & Video Processing
- Auto-captions images using the **BLIP model**.
- S3-compatible integration to load **images, videos, and captions**.

### Supported Files
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


Discover and enjoy our app, [AgrorithmFarm](https://agrorithm-farm.streamlit.app/)!
