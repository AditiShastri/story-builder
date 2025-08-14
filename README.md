
### Story Builder

This project is a web application that leverages a large language model to generate stories based on user input.

### Project Setup and Execution

This guide provides a step-by-step process to set up and run the `story-builder` project.

#### Prerequisites

  * Python 3.x
  * Git

-----

### **Important Notes Before You Begin**

  * **Installation Time**: Installing the dependencies from `requirements.txt` may take several minutes.
  * **First-Time Model Download**: The very first time you run the application, the OFA-Sys/small-stable-diffusion-v0 image generation model will be downloaded. This is a large file and will take a significant amount of time depending on your internet speed. Subsequent runs will be much faster as the model will be cached.
  * **Response Time**: Generating a story using an AI model can be a time-consuming process. Please be patient while waiting for a response, especially if you're using a CPU to generate images.

-----

#### 1\. Clone the repository and set up a virtual environment

First, clone the project from its repository and navigate into the project directory. It is highly recommended to use a virtual environment to manage dependencies.

```bash
git clone https://github.com/AditiShastri/story-builder.git
cd story-builder
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts/activate`
```

#### 2\. Install dependencies

Install all required packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

#### 3\. Configure your Hugging Face API token

The project requires access to the Hugging Face Hub to use the story generation model. You will need to create a `.env` file in the project's root directory to store your API token.

1.  **Get your token**:

      * Sign up for or log in to Hugging Face.
      * Go to your profile settings and find "Access Tokens."
      * Generate a new fine grained token with "read" permissions and "inference" permissions.
      * <img width="1822" height="874" alt="Screenshot 2025-08-14 214612" src="https://github.com/user-attachments/assets/e0ac1dcc-05b2-4692-84f5-3adff95e1c2b" />

      * **Crucial**: Ensure you have agreed to share your contact information to access the `"mistralai/Mistral-7B-Instruct-v0.3"` model. This step is necessary to use the model.
      * <img width="1532" height="527" alt="Screenshot 2025-08-14 214543" src="https://github.com/user-attachments/assets/0bb79282-cf1f-4bc0-8c5f-988883c74e8c" />


2.  **Create `.env` file**:
    Create a file named `.env` in the root directory (the same folder as `manage.py`) and add the following line, replacing `"your_huggingface_token"` with your actual token:

    ```ini
    HUGGINGFACEHUB_API_TOKEN = "your_huggingface_token"
    ```

#### 4\. Prepare the database

Run the following commands to create the necessary database tables and apply any pending migrations.

```bash
python manage.py makemigrations
python manage.py migrate
```

#### 5\. Run the development server

Start the application by running the development server.

```bash
python manage.py runserver
```

The application should now be running on `http://127.0.0.1:8000/`. You can open this URL in your web browser to start using the story builder.
