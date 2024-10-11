# DocGPT 📚💬

DocGPT is an interactive question-answering tool designed to extract precise answers from uploaded PDF files. With DocGPT, users can effortlessly upload PDF documents and inquire about specific content within them, receiving accurate responses pinpointing the location of the relevant text.

## Setup Instructions 🛠️

1. **Clone Repository**: Begin by cloning this repository to your local system.

2. **Create Conda Environment** (optional but recommended): Set up a Conda environment for isolating dependencies.

3. **Install Dependencies**: Install the necessary Python packages using the following command:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Settings**: Adjust the settings according to your preferences in the `setup.yaml` file. The key parameters to focus on are `LLM_BASE_URL` and `MODEL_NAME`. While other parameters are pre-tuned for optimal performance, these two are crucial for specifying the location and the name of the language model. If you're using a locally hosted language model, ensure that its API is compatible with OpenAI's (a drop-in replacement for OpenAI).

5. **Provide API Token (if applicable)**: If you're using OpenAI or an external language model provider, provide the API token in the `secret.env` file. If you're using a locally hosted language model, put `empty` string as the api-key. Remember not to leave the API token field blank to avoid errors.

6. **Run the Application**: Execute the following command to launch the application:
    ```bash
    streamlit run streamlit_app.py
    ```
   By default, the app will be hosted at port `3040`. If you wish to customize the hosting settings, refer to the configuration file located inside the `.streamlit` directory.

## Usage 🚀

1. **Upload PDF**: Start by uploading the PDF document you want to query.
2. **Ask Questions**: Enter your questions in the provided text box.
3. **Get Answers**: DocGPT will analyze the uploaded PDF and provide precise answers along with the location of the text used to generate the response.


---

📌 For additional information or bug reports, please open an issue. Feedback is welcome!
