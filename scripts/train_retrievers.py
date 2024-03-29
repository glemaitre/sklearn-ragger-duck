# %% [markdown]
# # Training the retriever
#
# This notebook will train the lexical and semantic retriever and store them.
#
# Not bothering with a installable package for now. For now, important the modules
# directly from the source code.

# %%
import logging
import os
import sys
from pathlib import Path

import configuration as config
import joblib

sys.path.append(str(Path(__file__).parent.parent))
API_DOC = Path(config.API_DOC_PATH)
USER_GUIDE_DOC = Path(config.USER_GUIDE_DOC_PATH)
DEVICE = os.getenv("DEVICE", "cpu")

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# Define the training pipeline that extract the text chunks from the API documentation
# and then embed them using a sentence transformer.

# %%
from sklearn.pipeline import Pipeline

from ragger_duck.embedding import SentenceTransformer
from ragger_duck.retrieval import SemanticRetriever
from ragger_duck.scraping import APINumPyDocExtractor

embedding = SentenceTransformer(
    model_name_or_path="thenlper/gte-large",
    cache_folder=config.CACHE_PATH,
    device=DEVICE,
)
api_scraper = APINumPyDocExtractor()
pipeline = Pipeline(
    steps=[
        ("extractor", api_scraper),
        ("semantic_retriever", SemanticRetriever(embedding=embedding, top_k=15)),
    ]
)
pipeline.fit(API_DOC)
pipeline

# %% [markdown]
# Save the semantic retriever to be used in the inference time.

# %%
joblib.dump(
    pipeline.named_steps["semantic_retriever"], config.API_SEMANTIC_RETRIEVER_PATH
)

# %% [markdown]
# Create a lexical retriever to match some keywords. We take a very long chunk to be
# sure that the keywords are present in the chunk.

# %%
from sklearn.feature_extraction.text import CountVectorizer

from ragger_duck.retrieval import BM25Retriever

count_vectorizer = CountVectorizer(ngram_range=(1, 5))
pipeline = Pipeline(
    steps=[
        ("extractor", api_scraper),
        (
            "lexical_retriever",
            BM25Retriever(count_vectorizer=count_vectorizer, top_k=15),
        ),
    ]
).fit(API_DOC)
pipeline

# %% [markdown]
# Save the lexical retriever to be used in the inference time.

# %%
joblib.dump(
    pipeline.named_steps["lexical_retriever"], config.API_LEXICAL_RETRIEVER_PATH
)

# %%
from ragger_duck.scraping import UserGuideDocExtractor

embedding = SentenceTransformer(
    model_name_or_path="thenlper/gte-large",
    cache_folder=config.CACHE_PATH,
    device=DEVICE,
)
user_guide_scraper = UserGuideDocExtractor(chunk_size=700, chunk_overlap=10, n_jobs=-1)
pipeline = Pipeline(
    steps=[
        ("extractor", user_guide_scraper),
        ("semantic_retriever", SemanticRetriever(embedding=embedding, top_k=15)),
    ]
)
pipeline.fit(USER_GUIDE_DOC)
pipeline

# %%
joblib.dump(
    pipeline.named_steps["semantic_retriever"],
    config.USER_GUIDE_SEMANTIC_RETRIEVER_PATH,
)

# %%
from sklearn.feature_extraction.text import CountVectorizer

from ragger_duck.retrieval import BM25Retriever

count_vectorizer = CountVectorizer(ngram_range=(1, 5))
user_guide_scraper = UserGuideDocExtractor(chunk_size=700, chunk_overlap=10, n_jobs=-1)
pipeline = Pipeline(
    steps=[
        ("extractor", user_guide_scraper),
        (
            "lexical_retriever",
            BM25Retriever(count_vectorizer=count_vectorizer, top_k=15),
        ),
    ]
).fit(USER_GUIDE_DOC)
pipeline

# %%
joblib.dump(
    pipeline.named_steps["lexical_retriever"], config.USER_GUIDE_LEXICAL_RETRIEVER_PATH
)

# %%
