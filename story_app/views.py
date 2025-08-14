import os
import logging
from django.shortcuts import render
from django.views import View
from django.views.generic import ListView
from dotenv import load_dotenv

# Import your model and the AI services, including the custom exception
from .models import Story
from .services import InappropriatePromptError # Import the custom exception
from .chain import get_story_builder_chain # Import the new chain builder

# Load environment variables (like your HUGGINGFACEHUB_API_TOKEN)
load_dotenv()

class GeneratorView(View):
    """
    Handles the GET request for showing the form and the POST request
    for running the AI story generation pipeline.
    """
    template_name = 'story_app/index.html'
    result_template_name = 'story_app/result.html'
    
    def get(self, request, *args, **kwargs):
        """Displays the initial story generation form."""
        return render(request, self.template_name)

    def post(self, request, *args, **kwargs):
        """Handles the form submission and runs the full AI pipeline."""
        user_prompt = request.POST.get('prompt_text')
        
        # --- Input Validation ---
        if not user_prompt:
            return render(request, self.template_name, {'error': 'Prompt cannot be empty.'})

        context = {'user_prompt': user_prompt}

        try:
            # --- Initialize and Run the Full LangChain Chain ---
            logging.info("Initializing the full Story Builder Chain...")
            story_chain = get_story_builder_chain()
            
            logging.info(f"Invoking chain with prompt: '{user_prompt}'")
            # The chain is invoked with a single call, passing the user prompt in a dictionary
            final_result = story_chain.invoke({"user_prompt": user_prompt})

            # --- Unpack the results from the chain's output dictionary ---
            package = final_result.get('package')
            image_prompts = final_result.get('prompts')
            final_image_path = final_result.get('final_image_path')

            if not all([package, image_prompts, final_image_path]):
                raise Exception("The AI pipeline failed to return a complete result. One or more steps may have failed.")

            # --- Save the complete result to the database ---
            story_instance = Story.objects.create(
                user_prompt=user_prompt,
                story_text=package.story,
                character_description=package.character_description,
                background_description=package.background_description,
                art_style=package.art_style_and_mood,
                final_character_prompt=image_prompts.character_prompt,
                final_background_prompt=image_prompts.background_prompt,
                combined_image_path=final_image_path,
            )

            context['story'] = story_instance

        except InappropriatePromptError as e:
            # Catch the specific safety refusal error
            logging.warning(f"Safety guardrail triggered for prompt: '{user_prompt}'")
            context['error'] = str(e) # Display the AI's refusal message
        
        except Exception as e:
            # Catch any other general errors during the pipeline
            logging.error(f"An unexpected error occurred in the generation pipeline: {e}", exc_info=True)
            context['error'] = f"An unexpected error occurred: {e}"

        # --- Render the final result page, showing either the story or an error ---
        return render(request, self.result_template_name, context)

class StoryGalleryView(ListView):
    """
    Displays a gallery of all previously generated stories.
    """
    model = Story
    template_name = 'story_app/gallery.html'
    context_object_name = 'stories'
    ordering = ['-created_at'] # Show the newest stories first
    paginate_by = 9 # Show 9 stories per page
