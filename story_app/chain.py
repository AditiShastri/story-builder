import os
import logging
from dotenv import load_dotenv

# Import the core logic functions from our services file
from .services import (
    generate_creative_package,
    regenerate_character_description, # Import the new fallback function
    engineer_final_prompts,
    generate_and_combine_images,
    CreativePackage,
    ImagePrompts
)

# Import LangChain components for building the chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define the Input Schema for our Chain ---
class StoryBuilderInput(BaseModel):
    user_prompt: str = Field(description="The user's initial creative idea for the story and scene.")

# --- The Main Chain Definition ---
def get_story_builder_chain():
    """
    Assembles and returns the full, invokable LangChain pipeline,
    now with a conditional fallback for character generation.
    """
    load_dotenv()
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # --- MODIFICATION: Using the more reliable v0.3 model ---
    REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    if not HUGGINGFACEHUB_API_TOKEN:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")

    # Define the condition for our branch: does a valid character description exist?
    has_character_description = RunnableLambda(
        lambda x: x.get("package") and x["package"].character_description and x["package"].character_description != "N/A"
    )

    # --- MODIFICATION: Using RunnableBranch for conditional logic ---
    full_chain = (
        RunnablePassthrough.assign(
            # Step 1: Generate the Creative Package
            package=RunnableLambda(
                lambda x: generate_creative_package(x['user_prompt'], REPO_ID, HUGGINGFACEHUB_API_TOKEN)
            ).with_config(run_name="GenerateCreativePackage")
        )
        | RunnableBranch(
            # If has_character_description is True, do nothing (passthrough).
            (has_character_description, RunnablePassthrough()),
            # If False, run the fallback function to regenerate the description.
            RunnableLambda(
                lambda x: regenerate_character_description(x, REPO_ID, HUGGINGFACEHUB_API_TOKEN)
            ).with_config(run_name="RegenerateCharacterDescription_Fallback")
        )
        | RunnablePassthrough.assign(
            # Step 2: Engineer the Final Prompts
            prompts=RunnableLambda(
                lambda x: engineer_final_prompts(x['package'], REPO_ID, HUGGINGFACEHUB_API_TOKEN)
            ).with_config(run_name="EngineerFinalPrompts")
        )
        | RunnablePassthrough.assign(
            # Step 3: Generate and Combine Images
            final_image_path=RunnableLambda(
                lambda x: generate_and_combine_images(x['prompts'])
            ).with_config(run_name="GenerateAndCombineImages")
        )
    )

    return full_chain.with_types(input_type=StoryBuilderInput)

# --- Example of How to Run the Chain (for testing) ---
if __name__ == "__main__":
    logging.info("Initializing the Story Builder Chain...")
    story_chain = get_story_builder_chain()
    prompt = "A stoic robot librarian in a vast, ancient library made of living, bioluminescent wood."
    logging.info(f"Invoking chain with prompt: '{prompt}'")
    final_result = story_chain.invoke({"user_prompt": prompt})

    logging.info("\n---  FINAL CHAIN OUTPUT ---")
    if final_result:
        package: CreativePackage = final_result.get('package')
        prompts: ImagePrompts = final_result.get('prompts')
        image_path = final_result.get('final_image_path')
        if all([package, prompts, image_path]):
            print("\n**STORY:**", package.story)
            print("\n**ART STYLE:**", package.art_style_and_mood)
            print("\n**ENGINEERED CHARACTER PROMPT:**", prompts.character_prompt)
            print("\n**ENGINEERED BACKGROUND PROMPT:**", prompts.background_prompt)
            print("\n**FINAL IMAGE PATH:**", image_path)
        else:
            print("The chain produced an incomplete result.")
    else:
        print("The chain failed to produce a result.")
    logging.info("\n--- Chain execution complete. ---")
