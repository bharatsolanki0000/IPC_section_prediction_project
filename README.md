# IPC Section Suggestion

A small Streamlit app that suggests relevant IPC sections for a crime description using sentence embeddings.

Quick start

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure `preprocess_data.pkl` exists in the project root. The repository includes a preprocessed pickle file. If not present, preprocess `FIR-DATA.csv` using the notebook `fir_project.ipynb`.

4. Run the Streamlit app:

```bash
streamlit run IPC.py
```

Notes

- The first run may download the sentence-transformers model and required NLTK data.
- If `preprocess_data.pkl` is missing, the app will show an error in the sidebar indicating the expected path.
