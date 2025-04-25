import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
from io import BytesIO

import streamlit as st
from st_files_connection import FilesConnection

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, ".."))
# file_path = os.path.join(base_dir, "Data", "Raw", "CAFBRain_Dataset", "images")
# caption_path = os.path.join(file_path, "Image Captions.json")


def load_image_captions(aws_path) -> Dict[str, str]:
    """
    Load image captions from JSON file.
    
    Returns:
        Dict[str, str]: A dictionary mapping image keys to their captions
    """
    # Initialize a dictionary to store captions
    image_captions_dict = {}
    conn = st.connection('s3', type=FilesConnection)
    # Load the JSON file
    try:
        image_caption_path = os.path.join(aws_path,"images/Image Captions.json")
        image_captions_dict = conn.read(image_caption_path, input_format="json", ttl=600)
        print("hi")
        
    except FileNotFoundError:
        print(f"Warning: Caption file not found at database")
    except json.JSONDecodeError:
        print(f"Warning: Error parsing JSON in database")

    return image_captions_dict

def search_images_by_caption(caption: str, image_captions: Dict[str, List[str]]) -> str:
    """
    Search for images using a caption by finding similar captions in the dictionary.
    
    Args:
        caption (str): The caption to search for
        image_captions_dict (Dict[str, List[str]]): Dictionary of image captions
        similarity_threshold (float): Minimum similarity score to consider a match
        
    Returns:
        List[str]: List of image keys that match the caption
    """
    try:
        # Initialize the sentence transformer model with a more stable version
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the search caption and ensure it's 2D (1 sample, n features)
        prompt_embedding = model.encode(caption, convert_to_tensor=True)
        
        # Find matches
        # Prepare captions and corresponding image filenames.
        captions = list(image_captions.values())
        filenames = list(image_captions.keys())

        # Compute embeddings for each candidate caption.
        caption_embeddings = model.encode(captions, convert_to_tensor=True)

        # Compute cosine similarity between the prompt and each caption.
        cosine_scores = util.cos_sim(prompt_embedding, caption_embeddings)[0]

        # Find the index of the caption with the highest similarity.
        best_index = int(cosine_scores.argmax())
        best_match_filename = filenames[best_index]
        
        return best_match_filename
    
    except Exception as e:
        print(f"Error in search_images_by_caption: {str(e)}")
        return []

def generate_image_from_prompt(prompt: str, client: Optional[OpenAI] = None, 
                             size: str = "1024x1024") -> Dict[str, str]:
    """
    Generate an image using DALL-E based on a prompt.
    
    Args:
        prompt (str): The prompt to generate the image from
        client (Optional[OpenAI]): OpenAI client instance
        size (str): Image size (default: "1024x1024")
        
    Returns:
        Dict[str, str]: Dictionary containing the image URL and prompt used
    """
    if client is None:
        client = OpenAI()
    
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        return image_url
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return {"image_url": "", "prompt": prompt}

def search_and_generate_images(image_caption: dict, search_query: str, generate_new: bool = False) -> Dict[str, List[str]]:
    """
    Search for existing images and optionally generate new ones based on a query.
    
    Args:
        search_query (str): The query to search for
        generate_new (bool): Whether to generate new images if no matches found
        
    Returns:
        Dict[str, List[str]]: Dictionary containing found and generated image URLs
    """
    print(base_dir)
    try:        
        # Generate new images if requested and no matches found
        if generate_new:
            try:
                image_url = generate_image_from_prompt(search_query)
                image_data = requests.get(image_url).content
                image = BytesIO(image_data)
                return all_captions, image
            except Exception as e:
                print(f"Error generating image: {e}")
        else:
            # Load image captions
            all_captions = image_caption
        
            # Search for matching images
            matches = search_images_by_caption(search_query, all_captions)
            
            # Delete the match from image captions file
            if matches:
                try:
                    
                    # Remove the matched entry
                    if matches in all_captions:
                        del all_captions[matches]
                    
                    print(f"Deleted {matches} from Image Captions.json")
                except Exception as e:
                    print(f"Error deleting match from captions file: {str(e)}")

            aws_path = st.secrets["AWS_path"]
            image_path = os.path.join(aws_path, "images/", matches)
            return all_captions, image_path

    except Exception as e:
        print(f"Error in search_and_generate_images: {str(e)}")
        return {"found_images": [], "generated_images": []} 