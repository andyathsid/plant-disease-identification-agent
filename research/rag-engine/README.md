# RAG Evaluation Pipeline for Plant Disease Retrieval

This project evaluates different retrieval strategies for a RAG (Retrieval-Augmented Generation) system focused on plant disease information.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment:**

    -   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**

    Create a `.env` file in the root of the project and add your API keys:

    ```
    OPENAI_API_KEY="your_openai_api_key"
    VOYAGE_API_KEY="your_voyage_api_key"
    JINA_API_KEY="your_jina_api_key"
    GOOGLE_API_KEY="your_google_api_key"
    ```

5.  **Run Qdrant:**

    Make sure you have a Qdrant instance running. You can use Docker:

    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```

## Usage

Open and run the `evaluation.ipynb` notebook to perform the evaluation. The notebook will:

1.  Load the corpus and QA datasets.
2.  Index the corpus using different embedding models.
3.  Run retrieval experiments with different models (vector search, hybrid search, rerankers).
4.  Evaluate the results using `ranx` and display the metrics.
