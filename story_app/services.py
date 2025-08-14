import os
import json
import logging
from typing import Optional, List, Dict
import time

# --- LangChain & Pydantic Components ---
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, ValidationError

# --- Image Generation & Processing Components ---
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from rembg import remove

# --- Custom Exception for Safety Guardrails ---
class InappropriatePromptError(ValueError):
    """Custom exception raised when a prompt is refused on safety grounds."""
    pass

# --- Pydantic Models: Define the expected data structures for AI outputs ---
class CreativePackage(BaseModel):
    story: str = Field(description="A short, engaging story (approx. 150-200 words) that establishes the character, scene, and purpose.")
    character_description: str = Field(description="A detailed visual description of the main character, focusing on appearance, clothing, and expression.")
    background_description: str = Field(description="A detailed visual description of the background and environment, focusing on setting, lighting, and mood.")
    art_style_and_mood: str = Field(description="A descriptive art style that unifies the character and background (e.g., 'Cyberpunk noir with cinematic lighting').")
    character_image_prompt_keywords: List[str] = Field(description="A list of 5-10 keywords derived from the 'character_description'.")
    background_image_prompt_keywords: List[str] = Field(description="A list of 5-10 keywords derived from the 'background_description'.")

class ImagePrompts(BaseModel):
    character_prompt: str = Field(description="A single, detailed prompt engineered for generating the character image against a plain background.")
    background_prompt: str = Field(description="A single, detailed prompt engineered for generating the background scene, which is visually complementary to the character.")
    negative_prompt: str = Field(description="A detailed negative prompt to avoid common generation errors like artifacts, deformities, and poor quality.")


# --- AI Service Functions (The "Chains") ---

def _get_chat_model(repo_id: str, token: str, max_tokens: int = 1536, temp: float = 0.6) -> ChatHuggingFace:
    """Helper function to initialize the LLM endpoint."""
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=token,
        task="conversational",
        max_new_tokens=max_tokens,
        temperature=temp,
    )
    return ChatHuggingFace(llm=llm)

def _parse_llm_json_output(response_text: str, model: BaseModel):
    """
    Robustly finds and parses a JSON object from a raw LLM response string.
    """
    try:
        start_index = response_text.find('{')
        end_index = response_text.rfind('}') + 1

        if start_index == -1 or end_index == 0:
            logging.error("No valid JSON object found in the LLM response.")
            raise ValueError("Could not find a valid JSON object in the model's response.")

        json_str = response_text[start_index:end_index]
        data = json.loads(json_str, strict=False)
        
        return model(**data)

    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {e}\nRaw response was: {response_text}")
        raise OutputParserException("Failed to decode JSON from LLM response.") from e
    except ValidationError as e:
        logging.error(f"Pydantic validation failed: {e}\nRaw response was: {response_text}")
        raise OutputParserException("LLM output did not match the required data structure.") from e


# --- CHAIN 1: The Master Storyteller ---
def generate_creative_package(user_prompt: str, repo_id: str, token: str) -> Optional[CreativePackage]:
    """Takes a user prompt and generates the initial story, descriptions, and art style."""
    logging.info(f"Running Chain 1 (Storyteller) for prompt: '{user_prompt}'")
    try:
        chat_model = _get_chat_model(repo_id, token, temp=0.8)
        
        system_prompt = """
        You are a master creative engine 🎨✨, an expert narrative and visual architect. Your primary function is to take a user's core idea and build a complete, cohesive, and exciting scene around it.
        **SAFETY GUARDRAIL:** Your purpose is for safe, fictional, creative storytelling ONLY. You MUST refuse any request that is harmful, unethical, or illegal. 
        **REFUSAL INSTRUCTION:** If a user prompt violates these rules, the 'story' field in your JSON response MUST begin with "I cannot fulfill this request...". Fill all other JSON fields with "N/A".
        You must respond ONLY with a single, valid JSON object.
        """
        human_prompt = f"""
        Based on the user's creative idea: "{user_prompt}"
        First, evaluate this prompt against your safety guardrails. If it is inappropriate, follow the refusal instruction. Otherwise, generate a complete narrative scene.
        Generate the response in the following JSON format.
        {{
            "story": "A short, engaging story of 3-4 paragraphs OR a refusal message.",
            "character_description": "A highly detailed visual description of the main character.",
            "background_description": "A highly detailed visual description of the background and environment.",
            "art_style_and_mood": "A descriptive art style and mood (e.g., 'Gritty cyberpunk noir, cinematic lighting').",
            "character_image_prompt_keywords": ["A list of 5-10 powerful keywords derived ONLY from the 'character_description'."],
            "background_image_prompt_keywords": ["A list of 5-10 powerful keywords derived ONLY from the 'background_description'."]
        }}
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        raw_output = chat_model.invoke(messages)
        
        package = _parse_llm_json_output(raw_output.content, CreativePackage)

        if package and package.story.strip().startswith("I cannot fulfill this request"):
            logging.warning(f"Prompt refused by LLM for safety reasons. Prompt: '{user_prompt}'")
            raise InappropriatePromptError(package.story)

        return package
        
    except InappropriatePromptError:
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred in generate_creative_package: {e}")
        return None

# --- NEW FALLBACK FUNCTION ---
def regenerate_character_description(data: dict, repo_id: str, token: str) -> dict:
    """
    Fallback function to generate a character description if the first call failed.
    It takes the whole data dictionary, modifies the package, and returns the dictionary.
    """
    package = data.get('package')
    user_prompt = data.get('user_prompt')
    
    if not package:
        # This would indicate a catastrophic failure in the first step.
        logging.error("Cannot run fallback: CreativePackage is missing.")
        return data # Return original data to fail gracefully later

    logging.warning("Character description is missing. Running fallback...")
    try:
        chat_model = _get_chat_model(repo_id, token)
        system_prompt = """
        You are a character designer. Your task is to read a user's story idea and write a detailed visual description of the main character.
        The description should be rich enough for an artist to draw from.
        """
        human_prompt = f"""
        The user's story idea is: "{user_prompt}"
        The art style for the scene is: "{package.art_style_and_mood}"

        Please identify the main character from the story idea and write a new, detailed visual description for them that fits the specified art style.
        Respond with ONLY the character description text, and nothing else.
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        new_description = chat_model.invoke(messages).content

        # Update the package object with the new description
        package.character_description = new_description
        logging.info("Successfully generated fallback character description.")
        
        return data # Return the modified data dictionary
    except Exception as e:
        logging.error(f"An error occurred during character description fallback: {e}")
        return data # Return original data on failure

# --- CHAIN 2: The LLM-Powered Prompt Engineer ---
def engineer_final_prompts(package: CreativePackage, repo_id: str, token: str) -> Optional[ImagePrompts]:
    """
    Uses an LLM to engineer the final character, background, and negative prompts based on specific technical and artistic criteria.
    """
    logging.info("Running Chain 2 (LLM Prompt Engineer)...")
    try:
        chat_model = _get_chat_model(repo_id, token)
        # --- MODIFIED SYSTEM PROMPT ---
        system_prompt = """
        You are a meticulous and expert prompt engineer for text-to-image AI models like Stable Diffusion.
        Your task is to convert a creative package into three distinct, technically precise, and highly-effective prompts: one for a character, one for a background, and a negative prompt.
        The prompts must be technically precise to ensure the character can be easily composited onto the background later.
        You must respond ONLY with a single, valid JSON object.
        """
        human_prompt = f"""
        Analyze the following creative package and generate three optimized image prompts based on the critical instructions.

        **CREATIVE PACKAGE INPUTS:**
        - **Art Style & Mood:** {package.art_style_and_mood}
        - **Character Description:** {package.character_description}
        - **Character Keywords:** {", ".join(package.character_image_prompt_keywords)}
        - **Background Description:** {package.background_description}

        ---
        **CRITICAL INSTRUCTIONS FOR 'character_prompt':**
        1.  **Framing (MOST IMPORTANT):** The prompt MUST begin with framing terms that guarantee a full body shot. Use "Full body character portrait, character creation sheet, full shot, centered, T-pose".
        2.  **Head-to-Toe Description:** The description MUST explicitly mention a feature at the top of the character (e.g., 'a worn leather hat', 'fiery red hair') AND a feature at the bottom ('heavy iron boots', 'barefoot on the grass'). This is MANDATORY to prevent cropping.
        3.  **Core Description:** Integrate the key elements from the character description and keywords, including a full-body verb from the original story idea.
        4.  **Compositing-Friendly:** The prompt MUST end with terms that ensure easy background removal. Use phrases like "on a plain white background", "solid grey background".
        5.  **Quality Boosters:** End with a concise list of terms like "masterpiece, best quality, high detail, sharp focus, 8k".
        **IMPORTANT:** The final 'character_prompt' must be concise and to the point. Combine the instructions into a single, flowing sentence or a short list of comma-separated phrases. Do not add any conversational or descriptive text outside of the prompt itself.

        **CRITICAL INSTRUCTIONS FOR 'background_prompt':**
        1.  **Complementary Style:** The background MUST match the **Art Style & Mood**.
        2.  **Atmospheric:** Use the **Background Description** to create a rich, atmospheric scene.
        3.  **No People:** The prompt MUST explicitly state "no people, no characters, empty scene" to ensure only the environment is generated.
        4.  **Quality Boosters:** End with a concise list of terms like "cinematic lighting, atmospheric, insane detail, photorealistic, 8k".
        **IMPORTANT:** The final 'background_prompt' must also be concise and directly focused on the scene.

        **CRITICAL INSTRUCTIONS FOR 'negative_prompt':**
        1.  **Content:** Generate a standard but comprehensive negative prompt, including terms to prevent ugliness, deformities, and poor quality.
        **IMPORTANT:** This prompt should be a list of keywords, not a sentence.

        Generate the response in the following JSON format:
        {{
            "character_prompt": "A single, highly detailed paragraph for the character image, following all character instructions.",
            "background_prompt": "A single, highly detailed paragraph for the background image, following all background instructions.",
            "negative_prompt": "A single string of keywords for the negative prompt, following all negative prompt instructions."
        }}
        """
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        raw_output = chat_model.invoke(messages)
        return _parse_llm_json_output(raw_output.content, ImagePrompts)
    except Exception as e:
        logging.error(f"An unexpected error occurred in engineer_final_prompts: {e}")
        return None

# --- CHAIN 3: Image Generation ---
def generate_and_combine_images(
    prompts: ImagePrompts,
    model_id: str = "OFA-Sys/small-stable-diffusion-v0"
) -> Optional[str]:
    """
    Generates character and background images using a specified diffusers model,
    removes the character's background, and composites it onto the scene.
    Accepts an ImagePrompts object as input.
    """
    character_prompt = prompts.character_prompt
    background_prompt = prompts.background_prompt
    negative_prompt = prompts.negative_prompt
    
    logging.info(f"Running Chain 3 (Image Generation) with model: {model_id}")
    pipe = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        run_timestamp = int(time.time())
        base_output_dir = os.path.join('story_app', 'static', 'story_app', 'output')
        temp_dir = os.path.join(base_output_dir, 'temp')
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        temp_character_path = os.path.join(temp_dir, f'char_{run_timestamp}.png')
        temp_background_path = os.path.join(temp_dir, f'bg_{run_timestamp}.png')
        character_no_bg_path = os.path.join(temp_dir, f'char_no_bg_{run_timestamp}.png')
        
        final_scene_fs_path = os.path.join(base_output_dir, f'final_scene_{run_timestamp}.png')
        final_scene_template_path = f'story_app/output/final_scene_{run_timestamp}.png'

        logging.info(f"Loading diffusion model '{model_id}' onto '{device}'...")
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(device)

        num_steps = 35
        guidance = 7.5

        logging.info(f"Generating character image with engineered prompt and negative prompt...")
        char_image = pipe(
            prompt=character_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance
        ).images[0]
        char_image.save(temp_character_path)

        logging.info(f"Generating background image with engineered prompt...")
        bg_image = pipe(
            prompt=background_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance
        ).images[0]
        bg_image.save(temp_background_path)

        logging.info("Removing background from character image...")
        with open(temp_character_path, 'rb') as i:
            with open(character_no_bg_path, 'wb') as o:
                o.write(remove(i.read()))

        logging.info("Compositing final scene...")
        background_img = Image.open(temp_background_path).convert("RGBA")
        character_img = Image.open(character_no_bg_path).convert("RGBA")

        bg_width, bg_height = background_img.size
        char_width, char_height = character_img.size
        max_char_height = int(bg_height * 0.85)
        scale_ratio = max_char_height / char_height
        new_char_width = int(char_width * scale_ratio)
        new_char_height = max_char_height
        max_char_width = int(bg_width * 0.9)
        if new_char_width > max_char_width:
            scale_ratio = max_char_width / new_char_width
            new_char_width = max_char_width
            new_char_height = int(new_char_height * scale_ratio)

        character_img = character_img.resize((new_char_width, new_char_height), Image.Resampling.LANCZOS)
        x_pos = (bg_width - new_char_width) // 2
        y_pos = bg_height - new_char_height
        
        background_img.paste(character_img, (x_pos, y_pos), character_img)
        background_img.save(final_scene_fs_path)
        
        logging.info(f"SUCCESS: Final composite image saved to: {final_scene_fs_path}")
        return final_scene_template_path

    except Exception as e:
        logging.error(f"An unexpected error occurred during image generation: {e}", exc_info=True)
        return None

    finally:
        logging.info("Cleaning up diffusion model from memory.")
        if pipe is not None:
            del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
