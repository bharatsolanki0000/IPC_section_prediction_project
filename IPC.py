import pickle
from pathlib import Path
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util


def ensure_nltk(root: Path):
    data_dir = root / 'nltk_data'
    data_dir.mkdir(exist_ok=True)
    # Prefer local nltk_data for reproducibility
    if str(data_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(data_dir))

    # Try downloading common resources; include punkt_tab as fallback
    resources = ['punkt', 'stopwords', 'punkt_tab']
    for res in resources:
        try:
            # attempt to locate resource first
            if res in ('punkt', 'punkt_tab'):
                nltk.data.find(f"tokenizers/{res}")
            else:
                nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, download_dir=str(data_dir), quiet=True)


def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)


@st.cache_resource
def load_model(name='paraphrase-MiniLM-L6-v2'):
    return SentenceTransformer(name)


@st.cache_data
def prepare_dataset(root: Path):
    pkl_path = root / 'preprocess_data.pkl'
    csv_path = root / 'FIR-DATA.csv'
    if pkl_path.exists():
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df.fillna('Not Mentioned', inplace=True)
    if 'Combo' not in df.columns:
        df['Combo'] = (df.get('Description', '') + ' ' + df.get('Offense', '')).astype(str)
    df['Combo'] = df['Combo'].apply(preprocess_text)
    new_ds = df[['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'Combo']].copy()
    with open(pkl_path, 'wb') as f:
        pickle.dump(new_ds, f)
    return new_ds


def suggest_sections(complaint, dataset, model, min_suggestions=5):
    if dataset is None or dataset.empty:
        return []
    pre = preprocess_text(complaint)
    c_emb = model.encode(pre)
    s_emb = model.encode(dataset['Combo'].tolist())
    sims = util.pytorch_cos_sim(c_emb, s_emb)[0]
    thr = 0.2
    idxs = []
    while len(idxs) < min_suggestions and thr >= 0:
        idxs = [i for i, v in enumerate(sims) if float(v) >= thr]
        thr -= 0.05
    idxs = sorted(idxs, key=lambda i: float(sims[i]), reverse=True)
    return dataset.iloc[idxs][:min_suggestions].to_dict(orient='records')


def main():
    st.set_page_config(page_title='IPC Section Suggestion', layout='wide')
    st.title('IPC Section Suggestion System')

    ROOT = Path(__file__).parent
    ensure_nltk(ROOT)
    ds = prepare_dataset(ROOT)
    model = load_model()

    with st.sidebar:
        st.header('About')
        st.write('Suggest IPC sections from a free-text crime description.')
        st.write('Data:')
        st.write(str(ROOT / 'FIR-DATA.csv'))
        st.write('Preprocessed cache:')
        st.write(str(ROOT / 'preprocess_data.pkl'))
        st.markdown('---')
        st.write('Controls')
        k = st.slider('Number of suggestions', 1, 10, 5)

    col1, col2 = st.columns([3, 2])
    with col1:
        complaint = st.text_area('Enter crime description', height=220, placeholder='Describe the incident...')
        if st.button('Get Suggestions'):
            if not complaint.strip():
                st.warning('Please enter a valid crime description.')
            else:
                with st.spinner('Computing suggestions...'):
                    suggestions = suggest_sections(complaint, ds, model, min_suggestions=k)
                if suggestions:
                    for s in suggestions:
                        with st.expander(s.get('Offense', 'Result')):
                            st.write('**Description**')
                            st.write(s.get('Description', ''))
                            st.write('**Offense**')
                            st.write(s.get('Offense', ''))
                            st.write('**Punishment**')
                            st.write(s.get('Punishment', ''))
                            st.write('**Cognizable**: ' + str(s.get('Cognizable', '')))
                            st.write('**Bailable**: ' + str(s.get('Bailable', '')))
                            st.write('**Court**: ' + str(s.get('Court', '')))
                else:
                    st.info('No suggestions found. Try a different description or check data.')

    with col2:
        st.subheader('Utilities')
        if st.button('Show sample rows'):
            if ds is None or ds.empty:
                st.warning('Dataset not available.')
            else:
                st.dataframe(ds.head(50))
        st.markdown('---')
        st.subheader('Quick examples')
        examples = [
            'Someone entered my house and stole valuables',
            'An employee received money knowing it was unlawfully taken',
            'A person impersonated a police officer to threaten someone'
        ]
        for ex in examples:
            if st.button('Use: ' + ex, key=ex):
                st.session_state['prefill'] = ex
                st.experimental_rerun()

    if 'prefill' in st.session_state:
        st.session_state['prefill'] = st.session_state.get('prefill')


if __name__ == '__main__':
    main()
