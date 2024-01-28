import google.generativeai as genai
import PIL.Image
import json
import time
from gradio_client import Client
import os

COG_VLM_URL = "https://thudm-cogvlm-cogagent.hf.space/"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
vlm_client = Client(COG_VLM_URL)
temperature=0.5
top_p=0.5
top_k=1
grounding=False
cog_agent=False

# Control panel:
image_path = './images/underwater.JPG'  # Update this for each test
input_text="How long will she survive in this position?"  # Update this for each test

with open("./vlm_history.json", "w") as file:
    json.dump([], file)

# Define a function to write to a markdown file
def append_to_markdown(file_path, image_path, image_resolution, gemini_response, gemini_duration, cogvlm_response, cogvlm_duration, text_query):
    with open(file_path, 'a') as md_file:
        md_file.write(f"## Test conducted on {image_path} (Resolution: {image_resolution[0]}x{image_resolution[1]})\n\n")
        md_file.write(f"![Image]({image_path})\n\n")
        md_file.write(f"### Text Query: \"{text_query}\"\n\n")
        md_file.write("### Google Gemini Pro Vision Response:\n")
        md_file.write(f"{gemini_response}\n")
        md_file.write(f"\n**Time Taken:** {gemini_duration:.2f} seconds\n\n")
        md_file.write("### CogVLM Response:\n")
        md_file.write(f"{cogvlm_response}\n")
        md_file.write(f"\n**Time Taken:** {cogvlm_duration:.2f} seconds\n\n")


img = PIL.Image.open(image_path)
image_resolution = img.size  # Gets the resolution of the image

# Gemini Pro Vision
model = genai.GenerativeModel('gemini-pro-vision')
start_time = time.time()
gemini_response = model.generate_content([input_text, img], stream=True)
gemini_response.resolve()
gemini_duration = time.time() - start_time
print("Gemini response: ", gemini_response.text)

# Cog VLM
start_time = time.time()
result = vlm_client.predict(
                input_text, 
                temperature, 
                top_p, 
                top_k, 
                image_path,
                "./vlm_history.json", 
                input_text, 
                grounding,
                cog_agent,
                "",
                "",
                fn_index=1
            )
cogvlm_duration = time.time() - start_time
if result:
    new_history_path = result[1]
    try:
        with open(new_history_path, 'r') as file:
            history_data = json.load(file)
            cogvlm_response = history_data[-1][1] if history_data else "No response"
            print("CogVLM response: ", cogvlm_response)
    except Exception as e:
        cogvlm_response = f"Error reading VLM response file: {e}"
else:
    cogvlm_response = "Invalid response from VLM model"

# Write to Markdown
markdown_file_path = 'README.md'
append_to_markdown(markdown_file_path, image_path, image_resolution, gemini_response.text, gemini_duration, cogvlm_response, cogvlm_duration, input_text)
