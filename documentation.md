
# Prompt Engineering Documentation
This document details the process followed to create prompts for story generation and image generation; along with an explanation of the prompts themselves.

The chain.py file orchestrates a langchain based chain which follows the flow given below. It calls functions defined in services.py
<img width="394" height="900" alt="Screenshot 2025-08-14 224551" src="https://github.com/user-attachments/assets/66dc3a2c-8160-4f8c-acec-72c87eeba414" />

## Step 1
1. How the Story Prompt Works
The core of the application's creative process begins with the "Master Storyteller" chain (generate_creative_package in services.py). This function takes a single user-provided text input and uses it to generate a CreativePackage Pydantic model.

It consists of a system prompt and a human prompt.
The system prompt for this LLM is created to establish a specific **persona** ("master creative engine, an expert narrative and visual architect") and enforce a strict JSON output format. This ensures the output is predictable and can be reliably parsed by the application to further generate the image prompts.

A guadrail is implemented to prohibit sexual and violent story generation. The format in which the LLM (in this case, Mistral) is expected to respond is clearly given to ensure predictable outputs. 

System Prompt:
    You are a master creative engine, an expert narrative and visual architect. Your job is to take a user's core idea and build a complete, cohesive, and       exciting scene around it.
    
    Output policy:
      Respond ONLY with a single, strict, valid JSON object.
      The JSON object MUST contain exactly these 6 fields and no others:
      "story" (string)
      "character_description" (string)
      "background_description" (string)
      "art_style_and_mood" (string)
      "character_image_prompt_keywords" (array of strings)
      "background_image_prompt_keywords" (array of strings)
      If you have no content for a field, use an empty string "" (for strings) or an empty array [] (for arrays).
      If the user's request is sexual or violent, return a valid JSON object with all 6 fields present, set "story" to "I cannot fulfill this request.", and set other fields to empty strings or empty arrays as appropriate. Do not include any other text.
      
      Formatting rules (strict):
      Use plain ASCII double quotes " for all JSON keys and string values.
      Escape any inner double quotes inside string values as: ".
      Escape newlines inside string values as \n. Do NOT insert raw newlines inside JSON string values.
      Do NOT use smart quotes, backticks, code fences, comments, or trailing commas.
      Arrays must be valid JSON arrays of strings (e.g., ["a","b","c"]).
      Do NOT include any fields beyond the 6 specified above.
      Do NOT include any text before or after the JSON object.
      
      Quality guidelines:
      Make "story" a single vivid scene with concrete sensory details and a clear moment of change.
      Keep descriptions consistent across fields; ensure keywords align with the character/background details.
      Keep each keywords array concise and visual (5–10 items), no full sentences.
      
      IMPORANT: Each string must start and end with double quotes, otherwise a hard failure is guaranteed.
      
      Schema template (structure only; replace values):
      {
      "story": "string with literal \n for line breaks",
      "character_description": "string",
      "background_description": "string",
      "art_style_and_mood": "string",
      "character_image_prompt_keywords": ["string","string","string"],
      "background_image_prompt_keywords": ["string","string","string"]
      }
      
      Notes:
      Your output will be parsed using a strict JSON parser (json.loads). Any deviation from the rules above will cause a hard failure—regenerate internally       to   ensure validity before responding.

The human prompt guides the LLM to use the user's idea to generate the story, character description, background description along with a few keywords that describe them. It also asks the LLM to create an art style and mood, so that the image prompts to result in cohesive character and background images.


Human Prompt:
    
    Based on the user's creative idea: "{user_prompt}"
    Generate the response in the following JSON format.
    {{
        "story": "A short, engaging story of 3-4 paragraphs expanding and enhancing the user's idea.",
        "character_description": "A highly detailed visual description of the main character based on the story generated. Include a single describing an action that the character is most likely performing at any given time.",
        "background_description": "A highly detailed visual description of the background and environment in which the character is present.",
        "art_style_and_mood": "A descriptive art style and mood (e.g., 'Gritty cyberpunk noir, cinematic lighting').",
        "character_image_prompt_keywords": ["A list of 5-10 powerful keywords derived ONLY from the 'character_description'."],
        "background_image_prompt_keywords": ["A list of 5-10 powerful keywords derived ONLY from the 'background_description'."]
    }}


    
### Example Iteration
User Prompt: "A stoic robot librarian in a vast, ancient library made of living, bioluminescent wood."

Result:

{
    "story": "Unit 734, its chassis a mosaic of rust and polished steel, glided silently between towering bookshelves. Each shelf was a living tree, its branches weaving into intricate patterns, glowing with a soft, bioluminescent light that pulsed in rhythm with the library's hum. It held a data-slate, its gears whirring faintly as it catalogued a recently found scroll. A faint scent of ozone and damp earth filled the air, a stark contrast to the sterile efficiency of the robot, yet it felt perfectly at home in this organic cathedral of knowledge.",
    "character_description": "A tall, slender robot, Unit 734, with a polished steel chassis that shows signs of rust and age. Its head is a simple sphere with a single, glowing blue optical sensor. Two articulated arms with three-fingered grippers extend from its torso, and its legs are designed for silent, efficient movement. It is holding a data-slate in its left hand, its posture stoic and purposeful. The robot is in a state of quiet contemplation as it processes information.",
    "background_description": "A vast, ancient library where the bookshelves are made of living, interconnected trees. The wood glows with a soft, pulsing bioluminescent light, casting long shadows across the floor. The air is thick with the scent of damp earth and old parchment. The library is silent except for a low, organic hum from the living walls and the faint whirring of gears from Unit 734.",
    "art_style_and_mood": "Steampunk with a touch of organic sci-fi, cinematic lighting, serene and contemplative mood.",
    "character_image_prompt_keywords": ["tall robot", "polished steel chassis", "rust patina", "glowing blue sensor", "three-fingered grippers", "stoic pose", "holding data-slate"],
    "background_image_prompt_keywords": ["vast ancient library", "living wood bookshelves", "bioluminescent light", "pulsing glow", "damp earth scent", "long shadows", "organic cathedral"]
}


## Step 2
A second LLM-powered chain (engineer_final_prompts in services.py) takes the result of step 1 and acts as a **"Prompt Engineer"**  to create three prompts  for the OFA-Sys/small-stable-diffusion-v0 model to generate images of the main character in the story and the background they are present in.

In this step, we leverage 3 prompts: a character prompt, negative prompt (for characteristics we do not want in the character image) and a background prompt.

1. Character Prompt Construction:
This  prompt for the character is based on the character_description, art_style_and_mood, and the character_image_prompt_keywords created in the previous step. It is verbose and contains a set of rules. The prompt \ must start with specific framing terms like "Full body character portrait, character creation sheet, full shot, centered, T-pose" to ensure the entire character is visible. This is crucial for avoiding cropped images and for later background removal.
It must explicitly include descriptive elements from the top of the head to the bottom of the feet to prevent the image model from cropping or cutting off parts of the body. In order to allow for composition with the background image, it  ends with terms like "on a plain white background" to make the background removal process. In addition, terms like "masterpiece, best quality, high detail, sharp focus, 8k" are used to improve the visual fidelity and distractors like accessories, objects, scenery, or props are explicitly prohibited to make sure that the image only contains the character.

2, Background Prompt Construction:
The background prompt is constructed from the background_description and art_style_and_mood. Its goal is to create an atmospheric scene that is visually complementary to the character. The prompt explicitly states "no people, no characters, empty scene" to ensure a clean background without any figures.
It uses descriptive text to create a rich environment. It also includes quality boosters similar to the character prompt, such as "cinematic lighting, atmospheric, insane detail, photorealistic, 8k".

3. Negative Prompt Construction:
Adding negative prompts significantly improved the quality of images generated. I noticed that the occurence of cropped hands and feet reduced after incorporating negative prompts into the workflow. It is created to guide the image model away from undesirable outputs. This is a list of keywords designed to prevent common generation errors like ugliness, deformities, artifacts, poor quality, and, critically, cropped head or missing feet, which reinforce the framing rules of the character prompt.

We achieve the three prompts detailed above using system and human prompts, much like in step 1.

System Prompt:

    You are a meticulous and expert prompt engineer for text-to-image AI models like Stable Diffusion.
    Your task is to convert a creative package into three distinct, technically precise, and highly-effective prompts: one for a character, one for a             background, and a negative prompt.
    The prompts must be technically precise to ensure the character can be easily composited onto the background later.
    You must respond ONLY with a single, valid JSON object.

Human Prompt:

    Analyze the following creative package and generate three optimized image prompts based on the critical instructions.
    
    **CREATIVE PACKAGE INPUTS:**
    - **Art Style & Mood:** {package.art_style_and_mood}
    - **Character Description:** {package.character_description}
    - **Character Keywords:** {", ".join(package.character_image_prompt_keywords)}
    - **Background Description:** {package.background_description}
    
    ---
    **CRITICAL INSTRUCTIONS FOR 'character_prompt':**
    1.  **Framing (MOST IMPORTANT):** The prompt MUST begin with framing terms that guarantee a full body shot. Head must be fully visible, and there must         be equal spacing above head and below feet. Use "Full body character portrait, character creation sheet, full shot, centered, T-pose". Ensure small f        figure, extra space above head and below feet.
    2.  **Head-to-Toe Description:** The description MUST explicitly mention a feature at the top of the character (e.g., 'a worn leather hat', 'fiery red         hair') AND a feature at the bottom ('heavy iron boots', 'barefoot on the grass'). This is MANDATORY to prevent cropping. 
    3.  **Core Description:** Integrate the key elements from the character description and keywords, including a full-body verb from the original story       idea r.
    4.  **Compositing-Friendly:** The prompt MUST end with terms that ensure easy background removal. Use phrases like "on a plain white background", "solid     grey background".
    5.  **Quality Boosters:** End with a concise list of terms like "masterpiece, best quality, high detail, sharp focus, 8k".
    6. ** Explicitly add phrases that prohibit any objects, props, scenery, or accessories that are not part of the described character.
      **IMPORTANT:** The final 'character_prompt' must be concise and to the point. Combine the instructions into a single, flowing sentence or a short list         of comma-separated phrases. Do not add any conversational or descriptive text outside of the prompt itself.
    
    **CRITICAL INSTRUCTIONS FOR 'background_prompt':**
    1.  **Complementary Style:** The background MUST match the **Art Style & Mood**.
    2.  **Atmospheric:** Use the **Background Description** to create a rich, atmospheric scene.
    3.  **No People:** The prompt MUST explicitly state "no people, no characters, empty scene" to ensure only the environment is generated.
    4.  **Quality Boosters:** End with a concise list of terms like "cinematic lighting, atmospheric, insane detail, photorealistic, 8k".
    **IMPORTANT:** The final 'background_prompt' must also be concise and directly focused on the scene.
    
    **CRITICAL INSTRUCTIONS FOR 'negative_prompt':**
    1.  **Content:** Generate a standard but comprehensive negative prompt, including terms to prevent ugliness, deformities, and poor quality
    .
    **IMPORTANT:** This prompt should be a list of keywords, not a sentence.
    
    Generate the response in the following JSON format:
    {{
        "character_prompt": "A single, highly detailed paragraph for the character image, following all character instructions.",
        "background_prompt": "A single, highly detailed paragraph for the background image, following all background instructions.",
        "negative_prompt": "A single string of keywords for the negative prompt, following all negative prompt instructions."
    }}



### Example Iteration


Engineered Character Prompt:
  
    "Full body character portrait, character creation sheet, full shot, centered, T-pose, a stoic robot librarian named Unit 734, with a polished steel           chassis that shows signs of rust and age. Its head is a simple sphere with a single, glowing blue optical sensor. Two articulated arms with three-          fingered grippers, legs designed for silent movement, holding a data-slate, polished boots, on a plain white background, masterpiece, best quality, high     detail, sharp focus, 8k, no accessories, no scenery, no props, no objects."


Engineered Background Prompt:
        
        "A vast, ancient library where the bookshelves are made of living, interconnected trees. The wood glows with a soft, pulsing bioluminescent light,       casting long shadows across the floor. The air is thick with the scent of damp earth and old parchment. Steampunk with a touch of organic sci-fi,             cinematic lighting, serene and contemplative mood, no people, no characters, empty scene, cinematic lighting, atmospheric, insane detail,                     photorealistic,         8k."


Negative Prompt:

    "ugly, deformed, disfigured, poor quality, low detail, extra limbs, fused fingers, blurry, bad anatomy, bad composition, artifacts, cropped head, head cut off, missing top of head, cropped feet, missing feet, zoomed in, feet cut off"


## References
1. https://www.qed42.com/insights/building-simple-effective-prompt-based-guardrails
2. https://medium.com/@zshariff70/langchain-simple-llm-chains-in-action-bda6950afc71
3. https://medium.com/@bijit211987/prompt-optimization-reduce-llm-costs-and-latency-a4c4ad52fb59
4. https://vinodveeramachaneni.medium.com/building-ai-agents-with-langchain-architecture-and-implementation-5e7fc3ccff88
5. https://readmedium.com/how-to-get-your-ai-art-generator-to-stop-cropping-off-peoples-heads-ae3183e79981
