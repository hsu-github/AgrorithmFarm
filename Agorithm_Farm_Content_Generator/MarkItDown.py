import os
import sys
from markitdown import MarkItDown
import openai
from getpass import getpass


def convert_file_to_markdown(client, input_filepath: str) -> str:
    """
    Convert the given file to Markdown using Microsoft MarkItDown.
    """
    try:
        # Initialize the converter (you can pass additional options if needed)
        converter = MarkItDown(llm_client=client, llm_model="gpt-4o-mini-2024-07-18",docintel_endpoint="<document_intelligence_endpoint>")
        
        # Convert the input file to Markdown text.
        markdown_text = converter.convert(input_filepath)
        return markdown_text

    except Exception as e:
        print(f"Error converting file {input_filepath}: {e}")
        sys.exit(1)

def write_markdown_to_file(markdown_text: str, output_filepath: str) -> None:
    """
    Write the markdown text to a file.
    """
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"Markdown successfully written to {output_filepath}")
    except Exception as e:
        print(f"Error writing Markdown to file: {e}")
        sys.exit(1)

def main():

    # 🔑 Initialize OpenAI API
    api_key = getpass("🔐 Enter your OpenAI API Key: ")
    client = openai.OpenAI(api_key=api_key)

    input_filepath = input("Enter the input file path: ")
    markdown_output = convert_file_to_markdown(client, input_filepath)
    
    # Print the resulting Markdown text
    print("Markdown Output:\n")
    print(markdown_output)

    base, _ = os.path.splitext(input_filepath)
    output_filepath = base + ".md"

    # Write the markdown output to a file.
    write_markdown_to_file(markdown_output.text_content, output_filepath)

if __name__ == "__main__":
    main()
