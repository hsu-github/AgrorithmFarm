import json
import os
import re
import cv2
from PIL import Image
from langchain_core.documents import Document
import warnings
import io
import numpy as np

import streamlit as st
from st_files_connection import FilesConnection
from tqdm import tqdm

def _read_images(RESDIR_PATH):
    """
    Reads all PNG images from the 'Raw/CAFBRain_Dataset/images' directory 
    and stores them in a dictionary without resizing.

    Returns:
    dict: A dictionary where keys are filenames and values are original images as NumPy arrays.
    """
    conn = st.connection('s3', type=FilesConnection)

    images_dict = {}
    try:
        # List all PNG files in the S3 bucket
        image_files = conn.fs.glob(os.path.join(RESDIR_PATH, "images/*.png"))
        print("get image file")
        

        for image_path in tqdm(image_files, desc="Loading images", unit="image"):
            # Extract just the filename from the path
            filename = os.path.basename(image_path)
            
            # Open the file as bytes
            with conn.open(image_path, "rb") as f:
                image_bytes = f.read()
                
            # First read with PIL to handle color profiles
            img_pil = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if not already
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            # Convert PIL image to numpy array
            img_np = np.array(img_pil)
            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Add to dictionary
            images_dict[filename] = img
    except Exception as e:
        print(f"Error loading images from S3: {str(e)}")

    return images_dict

def _convert_image_to_tokens(images_dict):
    """
    Generates captions for each image in the given dictionary using the BLIP model.

    Args:
    images_dict (dict): A dictionary where keys are filenames and values are image arrays (NumPy).

    Returns:
    dict: A dictionary where keys are filenames and values are image captions.
    """
    from transformers import BlipProcessor, BlipForConditionalGeneration
    # Initialize BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image_captions_dict = {}

    for filename, img_array in images_dict.items():
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb).convert("RGB")

        inputs = processor(images=img, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        image_captions_dict[filename] = caption

    return image_captions_dict

def _convert_jsonl_to_json(jsonl_files):
    """Convert a JSONL file to a JSON file."""
    conn = st.connection('s3', type=FilesConnection)
    
    try:
        # Check if file exists in S3
        try:
            with conn.open(jsonl_files, "r", encoding="utf-8") as infile:
                data = []
                for line in infile:
                    if isinstance(line, str) and line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"[ERROR] Failed to parse line in {jsonl_files}")
        except Exception as e:
            print(f"[ERROR] Failed to access {jsonl_files} in S3: {e}")
            return []

        return data  # Instead of writing to a file, return the parsed JSON data

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse {jsonl_files}: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return []

def _replace_filenames_with_images(image_filenames_dict, images_dict):
    """
    Replaces filenames in a dictionary with actual image data from images_dict.

    Parameters:
    image_filenames_dict (dict): A dictionary where values are lists of image filenames.

    Returns:
    dict: A dictionary with filenames replaced by actual image data.
    """
    updated_dict = {}
    
    for key, filenames in image_filenames_dict.items():
        updated_dict[key] = [images_dict.get(fname, None) for fname in filenames]  # Replace with actual images
    
    return updated_dict

def _read_collateral_json(RESDIR_PATH):
    """
    Reads 'Raw/CAFBRain_Dataset/text/collateral.jsonl' and converts it into an in-memory JSON structure.
    Constructs two dictionaries:
      - `collateral_text_dict`: Maps 'file_name' to extracted text.
      - `collateral_image_filenames_dict`: Maps 'file_name' to combined lists of base_images and extracted_infographics 
        (each sorted and deduplicated before merging).

    Returns:
    tuple: A tuple containing:
        - dict: `collateral_text_dict` where keys are filenames and values are extracted text content.
        - dict: `collateral_image_filenames_dict` where keys are filenames and values are lists of unique, sorted image filenames.
    """
    jsonl_path = os.path.join(RESDIR_PATH, "text", "collateral.jsonl")

    # Convert JSONL to JSON (in memory, no file written)
    data = _convert_jsonl_to_json(jsonl_path)
    if not data:
        print(f"[ERROR] Failed to load data from {jsonl_path}")
        return {}, {}

    # Construct dictionaries
    collateral_text_dict = {}  # Stores extracted text
    collateral_image_filenames_dict = {}  # Stores base_images + extracted_infographics (unique & sorted)

    for entry in data:
        file_name = entry.get("file_name")
        text_data = entry.get("text_data", [])
        base_images = sorted(set(entry.get("base_images", [])))  # Remove duplicates and sort base_images
        extracted_infographics = sorted(set(entry.get("extracted_infographics", [])))  # Remove duplicates and sort extracted infographics

        # Extract all text from "text_data" field
        text_content = "\n".join([item["text"] for item in text_data if "text" in item])

        # Merge the two lists after sorting and deduplication
        all_images = base_images + extracted_infographics

        if file_name:
            if file_name.endswith(".pdf"):
                file_name = file_name[:-4]

            collateral_text_dict[file_name] = text_content.strip()

            collateral_image_filenames_dict[file_name] = all_images  # Store sorted & unique images

    return collateral_text_dict, collateral_image_filenames_dict

def _read_powerpoints_json(RESDIR_PATH):
    """
    Reads 'Raw/CAFBRain_Dataset/text/powerpoints.jsonl' and converts it into an in-memory JSON structure.
    Constructs two dictionaries:
      - `powerpoints_text_dict`: Maps 'file_name' to extracted text, where all text entries are concatenated using '\n'.
      - `powerpoints_image_filenames_dict`: Maps 'file_name' to extracted images, keeping only the filename (not the full path).

    Returns:
    tuple: A tuple containing:
        - dict: `powerpoints_text_dict` where keys are filenames and values are extracted text content (concatenated with '\n').
        - dict: `powerpoints_image_filenames_dict` where keys are filenames and values are lists of extracted image filenames (only the filename, without path).
    """
    jsonl_path = os.path.join(RESDIR_PATH, "text", "powerpoints.jsonl")

    # Convert JSONL to JSON (in memory, no file written)
    data = _convert_jsonl_to_json(jsonl_path)
    if not data:
        print(f"[ERROR] Failed to load data from {jsonl_path}")
        return {}, {}

    # Construct dictionaries
    powerpoints_text_dict = {}  # Stores extracted text
    powerpoints_image_filenames_dict = {}  # Stores extracted image filenames

    for entry in data:
        file_name = entry.get("file_name")
        text_data = entry.get("text_data", [])
        images = entry.get("images", [])

        # Extract all text from "text_data" field
        text_content = "\n".join([item["text"] for item in text_data if "text" in item])

        # Extract only the filenames from the full image paths
        image_filenames = [os.path.basename(img) for img in images]

        if file_name:
            if file_name.endswith(".pptx"):
                file_name = file_name[:-5]

            powerpoints_text_dict[file_name] = text_content.strip()

            powerpoints_image_filenames_dict[file_name] = image_filenames  # Store image list

    return powerpoints_text_dict, powerpoints_image_filenames_dict

def _read_blog_posts_json(RESDIR_PATH):
    """
    Reads 'CAFBRain_Dataset/text/blog_posts.jsonl' and converts it into an in-memory JSON structure.
    Constructs two dictionaries:
      - blog_posts_text_dict: Maps unique 'title' (e.g., '001_My Title') to extracted content (first line removed if exists).
      - blog_posts_image_filenames_dict: Maps unique 'title' to extracted image_filenames.

    Handles duplicate titles by appending prefix 001_, 002_, etc.

    Returns:
    tuple: A tuple containing:
        - dict: blog_posts_text_dict where keys are sequential IDs combined with titles, values are extracted text content.
        - dict: blog_posts_image_filenames_dict where keys are sequential IDs combined with titles, values are lists of image filenames.
    """
    jsonl_path = os.path.join(RESDIR_PATH, "text", "blog_posts.jsonl")

    # Convert JSONL to JSON (in memory, no file written)
    data = _convert_jsonl_to_json(jsonl_path)
    if not data:
        print(f"[ERROR] Failed to load data from {jsonl_path}")
        return {}, {}

    # Determine the number of digits required
    total_documents = len(data)
    digits_needed = len(str(total_documents))

    # Construct dictionaries
    blog_posts_text_dict = {}  # Stores extracted content
    blog_posts_images_dict = {}  # Stores extracted image filenames

    for idx, entry in enumerate(data, start=1):
        # Generate zero-padded sequential ID prefixed to title (using underscore)
        title = entry.get("title", "untitled").strip()
        sequential_key = f"{str(idx).zfill(digits_needed)}_{title}"

        raw_content = entry.get("content", "")
        # Process content: Remove first line if there are multiple lines
        content = "\n".join(raw_content.split("\n")[1:]).strip() if "\n" in raw_content else raw_content.strip()

        image_filenames = entry.get("image_filenames", [])

        # Store extracted content
        blog_posts_text_dict[sequential_key] = content

        blog_posts_images_dict[sequential_key] = image_filenames  # Store image list

    return blog_posts_text_dict, blog_posts_images_dict

def _read_videos(RESDIR_PATH):
    """
    Reads all MP4 video files from the dataset's 'Raw/CAFBRain_Dataset/video' directory,
    and stores them in a dictionary with filenames as keys and OpenCV VideoCapture objects as values.

    Returns:
    dict: A dictionary where keys are filenames and values are OpenCV VideoCapture objects.
    """
    videos_path = os.path.join(RESDIR_PATH, "video")
    videos_dict = {
        f: cv2.VideoCapture(os.path.join(videos_path, f))  # Load video content directly
        for f in os.listdir(videos_path) if f.lower().endswith(".mp4")
    }
    return videos_dict

def _read_and_process_captions(RESDIR_PATH):
    """
    Reads all TXT caption files from the dataset's 'Raw/CAFBRain_Dataset/video/captions' directory,
    cleans WEBVTT tags and timestamps, splits the text into segments, and stores each segment
    as a Document with metadata in a dictionary.

    Returns:
    dict: A dictionary where keys are filenames and values are lists of Document objects.
    """
    captions_path = os.path.join(RESDIR_PATH, "video", "captions")
    captions_dict = {}

    for filename in os.listdir(captions_path):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(captions_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                raw_text = file.read().strip()
                # Clean the raw WEBVTT text
                cleaned_text = _clean_webvtt_text(raw_text)

                # Split the cleaned text into sentences (can be adjusted based on needs)
                chunks = _split_into_sentences(cleaned_text)

                # Store each chunk as a Document
                document_chunks = []
                for idx, chunk in enumerate(chunks):
                    if chunk.strip():  # Ensure non-empty content
                        document_chunks.append({
                            "page_content": chunk.strip(),
                            "metadata": {"filename": filename, "chunk_index": idx}
                        })

                captions_dict[filename] = document_chunks

    return captions_dict

def _clean_webvtt_text(text):
    """
    Cleans WEBVTT caption content by removing timestamps and <c> tags, and collapsing lines.

    Args:
    text (str): Raw WEBVTT caption text.

    Returns:
    str: Cleaned plain text content.
    """
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        # Remove lines that are timestamps or empty
        if re.search(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", line):
            continue
        if "WEBVTT" in line or line.strip() == "":
            continue

        # Remove <c> tags
        line = re.sub(r"</?c>", "", line)

        # Strip leading/trailing spaces
        cleaned_lines.append(line.strip())

    return " ".join(cleaned_lines)

def _split_into_sentences(text):
    """
    Splits a given cleaned caption text into sentences based on punctuation marks (e.g., '.', '!', '?').

    Args:
    text (str): Cleaned caption text.

    Returns:
    list: A list of sentence-like chunks.
    """
    # Use regular expressions to split based on periods, exclamation marks, or question marks followed by spaces
    return re.split(r'(?<=[.!?])\s+', text.strip())

def _read_youtube_posts_json(videos_dict, captions_dict, RESDIR_PATH):
    """
    Reads 'Raw/CAFBRain_Dataset/text/youtube_posts.jsonl' and converts it into an in-memory JSON structure.
    Constructs two dictionaries:
      - `youtube_posts_videos_dict`: Maps 'video_id' to extracted actual video data.
      - `youtube_posts_captions_dict`: Maps 'video_id' to formatted caption text, including: Title, Description, Actual caption data.
    
    Returns:
    tuple: A tuple containing:
        - dict: `youtube_posts_videos_dict` where keys are video IDs and values are extracted actual video data.
        - dict: `youtube_posts_captions_dict` where keys are video IDs and values are formatted caption text.
    """
    jsonl_path = os.path.join(RESDIR_PATH, "text", "youtube_posts.jsonl")

    # Convert JSONL to JSON (in memory, no file written)
    data = _convert_jsonl_to_json([jsonl_path])
    if not data:
        print(f"[ERROR] Failed to load data from {jsonl_path}")
        return {}, {}

    # Construct dictionaries
    youtube_posts_videos_dict = {}  # Stores extracted video data
    youtube_posts_captions_dict = {}  # Stores formatted caption text

    for entry in data:
        video_id = entry.get("video_id")
        title = entry.get("title", "").strip()
        description = entry.get("description", "").strip()
        video_file = os.path.basename(entry.get("video_file", ""))
        captions_file = os.path.basename(entry.get("captions_file", ""))

        if video_id:
            # Store video data
            youtube_posts_videos_dict[video_id] = videos_dict.get(video_file, None)

            # Store formatted captions
            original_captions = captions_dict.get(captions_file, "")
            formatted_caption = f"title: {title}\ndescription: {description}\n{original_captions}"
            youtube_posts_captions_dict[video_id] = formatted_caption.strip()

    return youtube_posts_videos_dict, youtube_posts_captions_dict

def _read_grant_proposals_json(RESDIR_PATH):
    """
    Reads 'Raw/CAFBRain_Dataset/text/grant_proposals.jsonl' and converts it into an in-memory JSON structure.
    Constructs two dictionaries:
      - `grant_text_dict`: Maps 'file_name' to extracted text, where all text entries are concatenated using '\n'.

    Returns:
    tuple: A tuple containing:
        - dict: `grant_text_dict` where keys are filenames and values are extracted text content (concatenated with '\n').
    """
    jsonl_path = os.path.join(RESDIR_PATH, "text", "grant_proposals.jsonl")

    # Convert JSONL to JSON (in memory, no file written)
    data = _convert_jsonl_to_json([jsonl_path])
    if not data:
        print(f"[ERROR] Failed to load data from {jsonl_path}")
        return {}, {}

    # Construct dictionaries
    grant_text_dict = {}  # Stores extracted text

    for entry in data:
        file_name = entry.get("file_name")
        text_data = entry.get("text_data", [])

        # Extract all text from "text_data" field
        text_content = "\n".join([item["text"] for item in text_data if "text" in item])

        # Remove unnecessary line breaks:
        # 1. Replace multiple newlines with a single one
        # 2. Remove line breaks within paragraphs
        text_content = re.sub(r'\n{2,}', '\n', text_content)  # keep single line break for real breaks
        text_content = re.sub(r'(?<!\n)\n(?!\n)', ' ', text_content)  # remove single line breaks that are not paragraph breaks

        if file_name:
            if file_name.endswith(".pdf"):
                file_name = file_name[:-4]

            grant_text_dict[file_name] = text_content.strip()

    return grant_text_dict

def _load_jsonl_to_documents(RESDIR_PATH):
    documents = []
    with open(RESDIR_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())  # Read each line as JSON and parse it
            document = Document(
                page_content=data["page_content"],  # Use 'page_content' from the JSON
                metadata=data["metadata"]  # Use 'metadata' from the JSON
            )
            documents.append(document)  # Add the Document object to the list
    return documents

def _data_processing(RESDIR_PATH):
    """
    Calls all  data processing functions to read and structure data from CAFBRain_Dataset.
    Returns a dictionary containing the processed data.
    
    Returns:
    dict: A dictionary with keys as variable names and values as the corresponding data.
    """
    
    collateral_text_dict, _ = _read_collateral_json(RESDIR_PATH)
    powerpoints_text_dict, _ = _read_powerpoints_json(RESDIR_PATH)
    blog_posts_text_dict, _ = _read_blog_posts_json(RESDIR_PATH)
    grant_text_dict = _read_grant_proposals_json(RESDIR_PATH)

    videos_dict = _read_videos(RESDIR_PATH)
    captions_dict = _read_and_process_captions(RESDIR_PATH)
    youtube_posts_videos_dict, youtube_posts_captions_dict = _read_youtube_posts_json(videos_dict, captions_dict, RESDIR_PATH)

    # Define return dictionary
    result_dict = {
        "collateral_text_dict": collateral_text_dict,
        "powerpoints_text_dict": powerpoints_text_dict,
        "blog_posts_text_dict": blog_posts_text_dict,
        "grant_text_dict": grant_text_dict,
        "youtube_posts_videos_dict": youtube_posts_videos_dict,
        "youtube_posts_captions_dict": youtube_posts_captions_dict
    }

    # Print keys in the required format
    print("All parameters:")
    for key in result_dict.keys():
        print(key)

    return result_dict

def _image_processing(RESDIR_PATH, image_captions=False):
    """
    Calls all  data processing functions to read and structure data from CAFBRain_Dataset.
    Returns a dictionary containing the processed data.
    
    Returns:
    dict: A dictionary with keys as variable names and values as the corresponding data.
    """
    # Read images from S3
    images_dict = _read_images(RESDIR_PATH)
    
    _, collateral_image_filenames_dict = _read_collateral_json(RESDIR_PATH)
    collateral_images_dict = _replace_filenames_with_images(collateral_image_filenames_dict, images_dict)
    
    _, powerpoints_image_filenames_dict = _read_powerpoints_json(RESDIR_PATH)
    powerpoints_images_dict = _replace_filenames_with_images(powerpoints_image_filenames_dict, images_dict)
    
    _, blog_posts_image_filenames_dict = _read_blog_posts_json(RESDIR_PATH)
    blog_posts_images_dict = _replace_filenames_with_images(blog_posts_image_filenames_dict, images_dict)

    if image_captions:
        image_captions_dict = _convert_image_to_tokens(images_dict)
        collateral_image_captions_dict = _replace_filenames_with_images(collateral_image_filenames_dict, image_captions_dict)
        powerpoints_image_captions_dict = _replace_filenames_with_images(powerpoints_image_filenames_dict, image_captions_dict)
        blog_posts_image_captions_dict = _replace_filenames_with_images(blog_posts_image_filenames_dict, image_captions_dict)
    else:
        collateral_image_captions_dict = ""
        powerpoints_image_captions_dict = ""
        blog_posts_image_captions_dict = ""
    
    # Define return dictionary
    result_dict = {
        "collateral_images_dict": collateral_images_dict,
        "collateral_image_captions_dict": collateral_image_captions_dict,
        "powerpoints_images_dict": powerpoints_images_dict,
        "powerpoints_image_captions_dict": powerpoints_image_captions_dict,
        "blog_posts_images_dict": blog_posts_images_dict,
        "blog_posts_image_captions_dict": blog_posts_image_captions_dict
        
    }

    # Print keys in the required format
    print("All parameters:")
    for key in result_dict.keys():
        print(key)

    return result_dict

def _image_processing_filenames(RESDIR_PATH):
    """
    Calls all data processing functions to read and structure data from CAFBRain_Dataset.
    Returns a dictionary containing the processed data.
    
    Returns:
    dict: A dictionary with keys as variable names and values as the corresponding data (filenames only).
    """
    _, collateral_image_filenames_dict = _read_collateral_json(RESDIR_PATH)
    
    _, powerpoints_image_filenames_dict = _read_powerpoints_json(RESDIR_PATH)
    
    _, blog_posts_image_filenames_dict = _read_blog_posts_json(RESDIR_PATH)

    # Define return dictionary
    result_dict = {
        "collateral_image_filenames_dict": collateral_image_filenames_dict,
        "powerpoints_image_filenames_dict": powerpoints_image_filenames_dict,
        "blog_posts_image_filenames_dict": blog_posts_image_filenames_dict,
    }

    # Print keys in the required format
    # print("All parameters:")
    # for key in result_dict.keys():
    #     print(key)

    return result_dict

