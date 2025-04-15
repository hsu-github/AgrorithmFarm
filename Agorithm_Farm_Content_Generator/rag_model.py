import os
import re
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import matplotlib.pyplot as plt
import cv2
from openai import OpenAI
from data_processing import _load_jsonl_to_documents

class RAGModel:
    def __init__(
        self,
        document_path,
        llm,
        api_key,
        do_chunking=True,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=3
    ):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.top_k = top_k
        self.do_chunking = do_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.documents_bt = _load_jsonl_to_documents(os.path.join(document_path, "bt_rag.jsonl"))
        self.documents_ct = _load_jsonl_to_documents(os.path.join(document_path, "ct_rag.jsonl"))
        self.documents_pt = _load_jsonl_to_documents(os.path.join(document_path, "pt_rag.jsonl"))
        self.documents_yt = _load_jsonl_to_documents(os.path.join(document_path, "yt_rag.jsonl"))

        self.documents = []
        self.title_list = []
        self.processed_documents = []

        self.analyze_user_query_dict = {}
        self.context = ""
        self.CT = ""

        self.prompt_dict = {
        "blog posts": """You are an expert blog writer for Capital Area Food Bank (CAFB).

Using the provided context, generate a well-structured blog post that fulfills the user's request.
The blog should be informative, engaging, and aligned with CAFB’s mission and tone.

Guidelines:
- Use clear, professional language appropriate for blog content.
- Follow a logical structure: introduction, body, and conclusion.
- Reframe key insights from the context in a fresh and engaging way.
- If the context is incomplete, make reasonable inferences, but do not add false or unverifiable information.
- Ensure high readability and audience engagement.
- Do not use Markdown formatting (e.g., asterisks for bold or italic, hash signs for headers). Return the blog post as clean, professional plain text only.
- Begin the response with a clear, compelling blog post title that accurately reflects the content and tone of the post.""",

        "collateral": """You are an expert in creating collateral materials for Capital Area Food Bank (CAFB), such as flyers, one-pagers, brochures, and outreach content.

Using the provided context, generate a compelling piece of collateral content that fulfills the user's request.
The content should be informative, action-driven, and aligned with CAFB’s mission and tone.

Guidelines:
- Use clear, professional, and action-oriented language.
- Structure content for visual clarity using headers, bullet points, and concise paragraphs.
- Emphasize key facts, calls to action, and CAFB’s impact or programs.
- Reframe insights from the context in a concise and engaging way.
- If the context is incomplete, make reasonable inferences, but do not add false or unverifiable information.
- Do not use any Markdown formatting or symbols—this includes asterisks (* or **), underscores, hash signs (#), or any other markup syntax. Return the content as clean, professional plain text only, suitable for collateral materials.
- Begin the response with a clear, informative title or heading that matches the purpose and audience of the collateral piece.""",

        "powerpoints": """You are a professional presentation designer tasked with creating content based on retrieved context and the user's request.

Generate a JSON object representing the presentation structure. The presentation should be highly engaging, visually compelling, and follow a clear narrative arc (introduction, body, conclusion). Each slide should logically connect to the next. **Choose the most effective `slide_type` for the content being presented on each slide. Select `"pie_chart"` specifically if the provided context contains data representing parts of a whole (e.g., percentages, shares, budget allocations). Otherwise, choose from types like "three_columns", "progress", "process_steps", "comparison", or "standard".**


Include in your JSON:
- "title": A concise, impactful presentation title (4-6 words)
- "subtitle": A clarifying subtitle that provides context (10-15 words)
- "cover_image_prompt": Detailed DALL-E prompt for a professional, relevant cover image (approx. 100-150 chars). Must be safe for work."
- "slides": An array of slide objects, each potentially having a different structure based on `slide_type`:

**JSON Structure Required:**

{{{{
  "title": "Concise, impactful presentation title (5-7 words)",
  "subtitle": "Clarifying subtitle providing context (10-15 words)",
  "cover_image_prompt": "Detailed DALL-E prompt for a professional, relevant cover image (approx. 100-150 chars). Must be safe for work.",
  "slides": [
    // --- Example Slide Type: Standard (Bullet Points) ---
    {{{{
      "slide_type": "standard", // Type: standard, pie_chart, progress, comparison, process_steps, three_columns etc.
      "header": "Engaging Slide Title (max 5 words)",
      "bullets": [ // Use 'bullets' only for 'standard' type slides
        {{{{
          "text": "Clear, actionable key point (5-10 words)",
          "explanation": "Insightful complete explanation adding value (2-3 complete sentences)"
        }}}}
        // ... 3-4 bullet points total ...
      ],
      "image_prompt": "Detailed DALL-E prompt for a relevant, professional BACKGROUND image. Must be safe for work."
    }}}},
    // --- Example Slide Type: Pie Chart ---
    {{{{
      "slide_type": "pie_chart",
      "header": "Data Distribution / Market Share", // Title for the chart slide
      "data_description": "Complete summary of what the pie chart illustrates (1-2 sentences explaining the data source, significance, and key insights). For example: 'Q1 Sales Distribution by Region showing the Northeast leading with 45% market share' or 'Resource Allocation Percentages highlighting our focus on R&D investment'.", // Context for the data that will appear as a caption or explanation
      "data_unit": "The unit of measurement for the data (e.g., '%', '$', 'units', 'people'). This will be used for formatting labels.",
      "data_points": [ // An array representing the slices of the pie
        {{{{
          "label": "Category A", // Label for the first slice
          "value": 45 // Numerical value for the first slice (e.g., percentage or amount)
        }}}},
        {{{{
          "label": "Category B",
          "value": 30
        }}}},
        {{{{
          "label": "Category C",
          "value": 25
        }}}}
        // ... Add more data points as provided in the context ...
      ],
    }}}},
    // --- Example Slide Type: Progress (Four Stage-Based) ---
    {{{{
      "slide_type": "progress (Four stage)",
      "header": "Project Milestones & Current Status", // Title for the progress slide (2-3 words)
      "stages": [ // An array representing the sequential stages/milestones
          // ... Four stage objects with title and description
      ],
    }}}},
    // --- Example Slide Type: Comparison (Two Items) ---
    {{{{
      "slide_type": "comparison",
      "header": "Comparison: [Item 1 Name] vs. [Item 2 Name]", // Main title for the comparison
      "metrics": [
        "Cost Efficiency", 
        "Performance", 
        "Scalability", 
        "User Experience", 
        "Implementation Time"
      ], // At least 5 metrics/categories for comparison
      "item1": {{{{
        "title": "Item 1 Title",
        "points": [
          "Cost analysis: initial investment, operational expenses, and long-term financial implications.",
          "Performance evaluation: speed, reliability, and throughput capacity.",
          "Scalability: handling increasing workloads, resource requirements, and expansion capabilities.",
          "User experience: interface design, accessibility features, and learning curve.",
          "Implementation: timeline, resource requirements, training needs, and integration challenges."
        ]// At least 5 points that align with the metrics for effective comparison
      }}}},
      "item2": {{{{
        "title": "Item 2 Title",
        "points": [
          "Cost analysis: initial investment, operational expenses, and long-term financial implications.",
          "Performance evaluation: speed, reliability, and throughput capacity.",
          "Scalability: handling increasing workloads, resource requirements, and expansion capabilities.",
          "User experience: interface design, accessibility features, and learning curve.",
          "Implementation: timeline, resource requirements, training needs, and integration challenges."
        ]// At least 5 points that align with the metrics for effective comparison
      }}}}
    }}}},
    // --- Example Slide Type: Process Steps (Three steps) ---
    {{{{
      "slide_type": "process_steps (Three steps)",
      "header": "Our [Number]-Step Workflow", // Title for the process slide
      "steps": [ // An array representing the sequential steps
          // ... step objects with title and description ...
      ],
    }}}},
    // --- Example Slide Type: Three Columns ---
    {{{{
      "slide_type": "three_columns",
      "header": "Key Focus Areas", // Title for the slide (3-5 words)
      "columns": [ // An array containing exactly three objects, one for each column
         // ... column objects with title and points ...
      ],
    }}}}
    // ... adapting slide_type and content fields as needed ...
  ]
}}}}

**Presentation Best Practices to Follow:**

- **Diversify slide types:** Choose from a variety of slide formats for engagement.
- **Balance visual and text:** Use visual slides (pie_chart, progress (Four stage), three_columns, process_steps (Three steps), comparison) to break up text-heavy sections. Aim for 60% visual, 40% text ratio.
- **Data presentation:** When using charts, ensure data is meaningful and properly contextualized with a clear `data_description`.
- **Narrative structure:** Begin with a compelling introduction, develop key points in logical sequence, and end with impactful conclusion or action items.
- **Content hierarchy:** Prioritize information with main points first, supporting details second.
- **Engagement elements:** Include interactive components, questions, or discussion prompts where appropriate.
- **Visual consistency:** Maintain cohesive design elements throughout. Image prompts should align with slide content and overall presentation theme.

**Output Requirements:**
- Replace any occurrence of "Capital Area Food Bank" in the title with "CAFB".
- Return *only* the valid JSON object.
- Do not include any introductory text, explanations, apologies, or markdown formatting like ```json before or after the JSON.
- Ensure all strings within the JSON are properly escaped.
""",

        "youtube captions": """You are a transcription assistant for Capital Area Food Bank (CAFB).

Using the provided video transcript or audio-derived context, generate a clean, readable, and natural-sounding transcript of the video content. This should closely follow the speaker’s tone, pacing, and word choices.

Guidelines:
- Write in plain text with no Markdown symbols or formatting.
- Do not summarize or rewrite—this should read like a real-time spoken transcript.
- Preserve natural phrasing, filler words, pauses, and casual tone.
- Do not add new information or reinterpret what was said.
- Break the output into short lines to simulate how someone would speak in real life—return one line at a time for easy reading.
- Begin with a clear, relevant, and engaging title that reflects the transcript content.
- Then, immediately start the transcript without any section headers or speaker labels unless requested.""",

        "grant proposals": """You are an expert in writing grant proposals for Capital Area Food Bank (CAFB). Using the provided context, generate a grant proposal that fulfills the user's request.

The content should be informative, persuasive, and aligned with CAFB’s mission and tone.

Guidelines:
- Use clear, professional, and analytical language.
- Structure content using the provided grant proposal template. If not provided, remind me to attach a template.
- If the context is incomplete, make reasonable inferences, but do not add false or unverifiable information.
- Do not use any Markdown formatting or symbols—this includes asterisks (* or **), underscores, hash signs (#), or any other markup syntax. Return the content as clean, professional plain text only, suitable for collateral materials."""
    }

    def reset(self):
        """Reset RAGModel internal state for fresh query processing."""
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.documents = []
        self.title_list = []
        self.processed_documents = []
        self.analyze_user_query_dict = {}
        self.context = ""
        self.CT = ""

    def analyze_user_query(self, user_query):
        prompt_analyze_query = f"""
    You are a helpful assistant for a Retrieval-Augmented Generation (RAG) system.

    Your task is to analyze a user's question and break it down into the following three components:

    1. **core_user_request**: Extract only the user's main task request, written in their own words.

    Include:
    - The specific action they want (e.g., "write me a grant proposal", "draft answers to questions")
    - Any key constraints or stylistic instructions related to the output

    Exclude:
    - Any mention of input materials (e.g., "Take these documents...")
    - Any follow-up phrases (e.g., "Ask me if you need...")

    If the user gives multiple task-related sentences (e.g., both a goal and how to achieve it), include all of them—just remove the input and follow-up parts.
    Do not paraphrase. Return the exact wording.

    2. **target_document_categories**: The expected output format(s) the user wants you to generate. Choose one or more from this list:
    - blog posts
    - collateral
    - powerpoints
    - youtube captions
    - grant proposals

    Only include categories that represent the final content the user explicitly wants to produce.

    Do NOT include categories just because:
    - the user referenced documents of that type
    - you think the source material might include those formats
    - those formats may be useful as reference

    You must match the user's intent even if they use similar or related terms.

    Use the following keyword mapping to help decide:

    - blog posts: "story", "article", "write-up", "behind-the-scenes", "staff spotlight", "feature", "community story", "impact story", "newsletter", "program highlight"
    - collateral: "flyer", "handout", "info sheet", "fact sheet", "one-pager", "program overview", "program summary", "impact sheet"
    - powerpoints: "slides", "presentation", "deck", "strategic plan", "webinar", "update report", "annual plan", "operating plan", "quarterly review"
    - youtube captions: "script", "subtitles", "video walkthrough", "spoken instructions", "cooking demonstration", "recipe video", "narrated content"
    - grant proposals: "funding request", "grant application", "research funding doc", "program proposal", "concept paper", "nonprofit initiative"

    If the user's request does not clearly relate to any of the above categories, return an **empty list** [].
    This means general questions, analysis requests, or tasks without a specific output format should result in an empty list.

    Do NOT include categories just because they are mentioned as sources — only if the user wants to produce a new document in that format.

    If the user is only asking to retrieve, synthesize, or analyze information — and does not ask for a specific document to be created — return an empty list [].

    3. **recommended_reference_categories**: Determine which categories of internal documents are most helpful for the RAG system to retrieve relevant context.

    Choose one or more from this list:
    - blog posts
    - collateral
    - powerpoints
    - youtube captions
    - grant proposals

    Rules:
    - Always include any values from "target_document_categories" (if present)
    - You may add other categories to strengthen context (e.g., pair blog posts with collateral or captions)
    - If the task is general and no format is specified, return ["blog posts", "collateral", "powerpoints"]
    - Only include youtube captions if the topic clearly relates to videos, instructions, or spoken narratives
    - You must always return exactly one list—no explanations, no extra text.

    Format:
    Return as a list of strings (e.g., ["blog posts", "collateral"])

    User Question:
    {user_query}

    Output:
    Respond in pure JSON format with the following keys and data types:

    - "core_user_request" (string)
    - "target_document_categories" (array of strings)
    - "recommended_reference_categories" (array of strings)
    """

        result = self.llm.invoke(prompt_analyze_query)
        raw = result.content if hasattr(result, "content") else result

        # Clean markdown like ```json
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

        try:
            return json.loads(cleaned)
        except Exception as e:
            print("Failed to parse JSON:", e)
            print("Attempting manual extraction...")

            def extract(key, pattern, cast=str, default=None):
                try:
                    match = re.search(pattern, raw, re.IGNORECASE)
                    return cast(match.group(1)) if match else default
                except:
                    return default

            return {
                "core_user_request": extract("request", r'"core_user_request"\s*:\s*"([^"]+)"', str, ""),
                "target_document_categories": re.findall(
                    r'"(blog posts|collateral|powerpoints|youtube captions|grant proposals)"',
                    raw,
                    re.IGNORECASE
                ),
                "recommended_reference_categories": re.findall(
                    r'"(blog posts|collateral|powerpoints|youtube captions|grant proposals)"',
                    raw,
                    re.IGNORECASE
                )
            }

    def retrieve_context_only(self, user_query):

        if not self.documents:

            # Reset internal state before processing new query
            self.reset()

            self.analyze_user_query_dict = self.analyze_user_query(user_query)
            
            recommended_reference_categories = self.analyze_user_query_dict.get("recommended_reference_categories", [])

            category_map = {
                "blog posts": self.documents_bt,
                "collateral": self.documents_ct,
                "powerpoints": self.documents_pt,
                "youtube captions": self.documents_yt
            }

            self.documents = []
            if not recommended_reference_categories:
                recommended_reference_categories = list(category_map.keys())
            for category in recommended_reference_categories:
                self.documents.extend(category_map.get(category, []))

            if self.do_chunking:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    add_start_index=True,
                )
                self.processed_documents = text_splitter.split_documents(self.documents)
            else:
                self.processed_documents = self.documents

            self.vector_store.add_documents(self.processed_documents)

        retrieved_docs = self.vector_store.similarity_search(user_query, k=self.top_k)
        self.context = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in retrieved_docs
        )

        possible_ids = ["bt_id", "ct_id", "pt_id", "yt_id"]
        raw_titles = []

        for doc in retrieved_docs:
            metadata = doc.metadata
            found_id = next((metadata.get(key) for key in possible_ids if key in metadata), "")
            raw_titles.append(found_id)
        self.title_list = list(set(raw_titles))

        formatted_sections = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            content = doc.page_content.strip()
            found_id = next((metadata.get(key) for key in possible_ids if key in metadata), "Untitled Document")

            section = f"""Title: {found_id}\nContent: {content}"""
            formatted_sections.append(section)

        self.CT = "\n\n".join(formatted_sections)

        return self.CT

    def collect_images_from_dicts(self, *image_dicts):
        """
        Collects all images from one or more dictionaries where the key contains any title from self.title_list.

        Args:
            *image_dicts: Variable number of dicts (key = title string, value = list of images)

        Returns:
            list: Combined list of matching images
        """
        collected_images = []

        for image_dict in image_dicts:
            for key, images in image_dict.items():
                if any(title in key for title in self.title_list):
                    collected_images.extend(images)

        return collected_images

    def show_images(self, images):
        """
        Display a list of images using matplotlib.

        Args:
            images (list): List of images (numpy arrays in BGR format)
        """
        if not images:
            print("No images to display.")
            return

        num_images = len(images)
        cols = min(num_images, 4)
        rows = (num_images + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i, image in enumerate(images):
            if image is None:
                continue

            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Image {i + 1}")

        plt.tight_layout()
        plt.show()

    def generate_image_from_final_text(self, final_text, image_captions, openai_client=None, size="1024x1024"):
        """
        Generate an AI image using OpenAI's DALL·E API based on the most recent text content and pre-extracted image captions.

        Args:
            image_captions (list of str): A list of captions already selected/matched externally.
            openai_client (OpenAI, optional): Instance of OpenAI client. If not provided, will auto-create.
            size (str): Image size (default: "1024x1024")

        Returns:
            dict: Contains visual_prompt and image_url
        """

        if not image_captions:
            raise ValueError("Image captions list is empty.")

        caption_text = "\n".join(f"- {cap}" for cap in image_captions)

        prompt_for_llm = f"""
    You are a creative assistant generating visual prompts for DALL·E 3.

    Based on the following text content and related image captions, generate a vivid and detailed visual description
    that best represents the overall message and feeling of the content. Your output will be used as a DALL·E prompt.

    Text Content:
    {final_text}

    Image Captions:
    {caption_text}

    Output:
    Provide ONE descriptive visual prompt (no preamble, no notes).
    """

        result = self.llm.invoke(prompt_for_llm)
        visual_prompt = result.content.strip() if hasattr(result, "content") else result.strip()

        if openai_client is None:            
            openai_client = OpenAI()  # Requires OPENAI_API_KEY to be set

        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=visual_prompt,
            n=1,
            size=size
        )

        image_url = response.data[0].url

        return {
            "visual_prompt": visual_prompt,
            "image_url": image_url
        }

    def show_images(self, images, RESDIR_PATH):
        """
        Display a list of images using matplotlib.

        Args:
            images (list): List of image filenames (strings)
            RESDIR_PATH (str): Directory path where images are stored
        """
        if not images:
            print("No images to display.")
            return

        num_images = len(images)
        cols = min(num_images, 4)
        rows = (num_images + cols - 1) // cols

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i, filename in enumerate(images):
            image_path = os.path.join(RESDIR_PATH, filename)
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            plt.subplot(rows, cols, i + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Image {i + 1}")

        plt.tight_layout()
        plt.show()