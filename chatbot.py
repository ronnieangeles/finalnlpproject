from pathlib import Path
import os
import random
import re
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# HYBRID OPEN-DOMAIN NLP CHATBOT
# ============================================================
#
# Architecture
# ------------
# 1. Customer-support questions:
#       TF-IDF + MiniLM semantic retrieval
#       -> top support examples
#       -> LLM grounded response (RAG-style)
#
# 2. General/open-domain questions:
#       LLM response with optional web search
#
# 3. If no OpenAI API key is configured:
#       customer-support retrieval still works,
#       but open-domain generation is unavailable.
#
# Training:
#       python chatbot.py --train
#
# Run:
#       streamlit run chatbot.py --server.fileWatcherType none
#
# ============================================================


# ------------------------------------------------------------
# PROJECT SETTINGS
# ------------------------------------------------------------

DATA_FILE = "twcs.csv"
ARTIFACT_DIR = Path("artifacts")

# Limit source rows so the project is practical on a laptop.
SOURCE_ROWS = 300_000
MAX_PAIRS = 20_000
AUGMENT_RATE = 0.25
RANDOM_SEED = 42


# ------------------------------------------------------------
# TF-IDF HYPERPARAMETERS
# ------------------------------------------------------------

MAX_FEATURES = 25_000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95


# ------------------------------------------------------------
# SENTENCE TRANSFORMER
# ------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------
# HYBRID RETRIEVAL SETTINGS
# ------------------------------------------------------------

SEMANTIC_WEIGHT = 0.70
LEXICAL_WEIGHT = 0.30
ENTITY_MATCH_BONUS = 0.08
ENTITY_MISMATCH_PENALTY = 0.04
TOP_K = 5

# Used to decide whether the question looks like a support request.
SUPPORT_ROUTE_THRESHOLD = 0.40

# Used only for the no-API-key fallback response.
RAW_RETRIEVAL_THRESHOLD = 0.50


# ------------------------------------------------------------
# OPEN-DOMAIN MODEL SETTINGS
# ------------------------------------------------------------

# Default OpenAI model. This can be overridden with an environment variable
# or with OPENAI_MODEL in Streamlit secrets.
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# TEXT AUGMENTATION
# ============================================================

SYNONYMS = {
    "help": ["assist", "support"],
    "problem": ["issue", "difficulty"],
    "issue": ["problem", "concern"],
    "need": ["require", "would like"],
    "fix": ["resolve", "correct"],
    "working": ["functioning", "operating"],
    "wrong": ["incorrect", "not right"],
    "slow": ["delayed", "sluggish"],
    "fast": ["quick", "rapid"],
    "message": ["response", "reply"],
    "service": ["support", "assistance"],
    "purchase": ["order"],
    "refund": ["reimbursement"],
}

FILLER_WORDS = {
    "really",
    "very",
    "just",
    "actually",
    "basically",
    "honestly",
}


# ============================================================
# SUPPORT ROUTING / ENTITY TERMS
# ============================================================

SUPPORT_CUES = {
    "not working",
    "doesn't work",
    "doesnt work",
    "won't work",
    "wont work",
    "won't turn on",
    "wont turn on",
    "can't log in",
    "cant log in",
    "cannot log in",
    "reset password",
    "forgot password",
    "locked out",
    "error",
    "issue",
    "problem",
    "broken",
    "charged",
    "charge",
    "billing",
    "refund",
    "order",
    "delivery",
    "cancelled",
    "canceled",
    "connection",
    "connect",
    "wifi",
    "account",
    "subscription",
    "payment",
    "service down",
    "app crashed",
    "app crashes",
    "crashing",
    "technical support",
    "customer service",
}

PRODUCT_TERMS = {
    "iphone",
    "ipad",
    "mac",
    "macbook",
    "android",
    "samsung",
    "windows",
    "xbox",
    "playstation",
    "router",
    "wifi",
    "email",
    "account",
    "password",
    "app",
    "phone",
    "tablet",
    "laptop",
    "internet",
    "order",
    "refund",
    "payment",
    "billing",
}


# ============================================================
# TEXT CLEANING
# ============================================================


def normalize_message(text):
    """Normalize text while preserving the user's meaning."""
    text = str(text)
    text = re.sub(r"https?://\S+", " <URL> ", text)
    text = re.sub(r"@\w+", " <USER> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def clean_response(text):
    """Clean social-media artifacts from a support response."""
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



def response_is_usable(raw_text, cleaned_text):
    """
    Filter responses that depend on missing links, private DMs,
    or too much outside context.
    """
    raw = str(raw_text).lower()
    cleaned = str(cleaned_text).lower().strip()

    if len(cleaned) < 25:
        return False

    bad_phrases = [
        "this article",
        "click here",
        "this link",
        "the link",
        "dm us",
        "send us a dm",
        "direct message us",
        "private message us",
        "follow us and dm",
        "send a private message",
    ]

    if any(phrase in cleaned for phrase in bad_phrases):
        return False

    # If a response depended heavily on a removed URL, skip short remnants.
    if re.search(r"https?://\S+", raw) and len(cleaned) < 80:
        return False

    return True


# ============================================================
# DATA AUGMENTATION
# ============================================================


def augment_message(text):
    """Create a light variation while attempting to preserve intent."""
    words = text.split()
    if not words:
        return text

    method = random.choice(["synonym", "filler"])

    if method == "synonym":
        choices = []
        for index, word in enumerate(words):
            key = word.lower().strip(".,!?;:")
            if key in SYNONYMS:
                choices.append((index, key))

        if choices:
            index, key = random.choice(choices)
            punctuation = words[index][-1] if words[index][-1:] in ".,!?;:" else ""
            words[index] = random.choice(SYNONYMS[key]) + punctuation
            return " ".join(words)

    removable = [
        index
        for index, word in enumerate(words)
        if word.lower().strip(".,!?;:") in FILLER_WORDS
    ]

    if removable:
        words.pop(random.choice(removable))
        return " ".join(words)

    return text


# ============================================================
# TWITTER DATASET PAIR BUILDING
# ============================================================


def first_response_id(value):
    """Extract the first response tweet id from response_tweet_id."""
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    return value.split(",")[0].strip()



def create_training_pairs():
    print("\nLoading Customer Support on Twitter dataset...")

    columns = [
        "tweet_id",
        "inbound",
        "text",
        "response_tweet_id",
    ]

    df = pd.read_csv(
        DATA_FILE,
        usecols=columns,
        nrows=SOURCE_ROWS,
        dtype={
            "tweet_id": "string",
            "response_tweet_id": "string",
        },
        low_memory=False,
    )

    print("Source rows loaded:", len(df))

    if df["inbound"].dtype != bool:
        df["inbound"] = (
            df["inbound"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

    lookup = (
        df.set_index("tweet_id")[["text", "inbound"]]
        .to_dict("index")
    )

    pairs = []
    inbound_rows = df[df["inbound"] == True]

    for _, row in inbound_rows.iterrows():
        response_id = first_response_id(row["response_tweet_id"])

        if not response_id or response_id not in lookup:
            continue

        response_record = lookup[response_id]

        # We only want an outbound/company response.
        if response_record["inbound"] != False:
            continue

        customer_message = normalize_message(row["text"])
        raw_response = str(response_record["text"])
        support_response = clean_response(raw_response)

        if len(customer_message) < 4:
            continue

        if not response_is_usable(raw_response, support_response):
            continue

        pairs.append((customer_message, support_response))

        if len(pairs) >= MAX_PAIRS:
            break

    pair_df = pd.DataFrame(pairs, columns=["message", "response"])
    pair_df = pair_df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    pair_df["source"] = "original"

    if pair_df.empty:
        raise RuntimeError(
            "No customer-response pairs were created. "
            "Verify twcs.csv or increase SOURCE_ROWS."
        )

    original_count = len(pair_df)
    print("Original training pairs:", original_count)

    augment_count = int(original_count * AUGMENT_RATE)
    selected = pair_df.sample(
        n=augment_count,
        random_state=RANDOM_SEED,
    )

    augmented_rows = []

    for _, row in selected.iterrows():
        changed = augment_message(row["message"])

        if changed != row["message"]:
            augmented_rows.append(
                {
                    "message": changed,
                    "response": row["response"],
                    "source": "augmented",
                }
            )

    if augmented_rows:
        pair_df = pd.concat(
            [pair_df, pd.DataFrame(augmented_rows)],
            ignore_index=True,
        )

    print("Augmented examples added:", len(pair_df) - original_count)
    print("Final training examples:", len(pair_df))

    return pair_df


# ============================================================
# TRAIN THE SUPPORT RETRIEVAL MODEL
# ============================================================


def train_chatbot():
    print("\n======================================")
    print("HYBRID NLP CHATBOT TRAINING")
    print("======================================")

    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"{DATA_FILE} was not found. Place twcs.csv in the same folder "
            "as chatbot.py."
        )

    ARTIFACT_DIR.mkdir(exist_ok=True)
    training_data = create_training_pairs()

    print("\nTraining TF-IDF model...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(training_data["message"])

    print("TF-IDF training complete.")
    print("TF-IDF feature count:", len(vectorizer.get_feature_names_out()))

    print("\nLoading Sentence Transformer...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating MiniLM semantic embeddings...")
    semantic_embeddings = encoder.encode(
        training_data["message"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    print("\nSaving model artifacts...")

    training_data.to_csv(
        ARTIFACT_DIR / "support_pairs.csv",
        index=False,
    )

    joblib.dump(
        vectorizer,
        ARTIFACT_DIR / "tfidf_vectorizer.joblib",
    )

    sparse.save_npz(
        ARTIFACT_DIR / "tfidf_matrix.npz",
        tfidf_matrix,
    )

    np.save(
        ARTIFACT_DIR / "semantic_embeddings.npy",
        semantic_embeddings,
    )

    print("\n======================================")
    print("TRAINING COMPLETE")
    print("======================================")
    print("Artifacts saved in:", ARTIFACT_DIR.resolve())
    print("\nRun the chatbot with:")
    print("streamlit run chatbot.py --server.fileWatcherType none")


# ============================================================
# ARTIFACT / MODEL LOADING
# ============================================================


def artifacts_exist():
    required = [
        ARTIFACT_DIR / "support_pairs.csv",
        ARTIFACT_DIR / "tfidf_vectorizer.joblib",
        ARTIFACT_DIR / "tfidf_matrix.npz",
        ARTIFACT_DIR / "semantic_embeddings.npy",
    ]

    return all(path.exists() for path in required)


@st.cache_resource
def load_support_resources():
    if not artifacts_exist():
        raise FileNotFoundError(
            "Support-model artifacts are missing. Run: python chatbot.py --train"
        )

    pairs = pd.read_csv(ARTIFACT_DIR / "support_pairs.csv")
    vectorizer = joblib.load(ARTIFACT_DIR / "tfidf_vectorizer.joblib")
    tfidf_matrix = sparse.load_npz(ARTIFACT_DIR / "tfidf_matrix.npz")
    semantic_embeddings = np.load(
        ARTIFACT_DIR / "semantic_embeddings.npy"
    ).astype("float32")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    return pairs, vectorizer, tfidf_matrix, semantic_embeddings, encoder


# ============================================================
# API KEY / OPENAI CLIENT
# ============================================================


def get_openai_model():
    """Return the configured OpenAI model name."""
    try:
        model = st.secrets["OPENAI_MODEL"]
        if model:
            return str(model)
    except Exception:
        pass

    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    # Also supports Streamlit Community Cloud secrets.
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


@st.cache_resource
def get_openai_client(api_key):
    if not api_key:
        return None

    return OpenAI(api_key=api_key, timeout=90.0)


# ============================================================
# SUPPORT RETRIEVAL
# ============================================================


def find_product_terms(text):
    lowered = text.lower()
    return {term for term in PRODUCT_TERMS if term in lowered}



def retrieve_support_examples(query, resources, top_k=TOP_K):
    (
        pairs,
        vectorizer,
        tfidf_matrix,
        semantic_embeddings,
        encoder,
    ) = resources

    normalized_query = normalize_message(query)

    query_tfidf = vectorizer.transform([normalized_query])
    lexical_scores = cosine_similarity(query_tfidf, tfidf_matrix).ravel()

    query_embedding = encoder.encode(
        [normalized_query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype("float32")

    semantic_scores = semantic_embeddings @ query_embedding

    combined_scores = (
        SEMANTIC_WEIGHT * semantic_scores
        + LEXICAL_WEIGHT * lexical_scores
    )

    # Add a small entity/product-aware re-ranking signal.
    query_entities = find_product_terms(normalized_query)

    if query_entities:
        for index, candidate in enumerate(pairs["message"]):
            candidate_entities = find_product_terms(str(candidate))

            if query_entities & candidate_entities:
                combined_scores[index] += ENTITY_MATCH_BONUS
            elif candidate_entities:
                combined_scores[index] -= ENTITY_MISMATCH_PENALTY

    count = min(top_k, len(combined_scores))
    top_indices = np.argsort(combined_scores)[-count:][::-1]

    results = []

    for index in top_indices:
        results.append(
            {
                "index": int(index),
                "score": float(combined_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "lexical_score": float(lexical_scores[index]),
                "message": str(pairs.iloc[index]["message"]),
                "response": str(pairs.iloc[index]["response"]),
            }
        )

    return results


# ============================================================
# ROUTER
# ============================================================


def has_support_cue(text):
    lowered = text.lower()

    if any(cue in lowered for cue in SUPPORT_CUES):
        return True

    # Common support-style constructions.
    support_patterns = [
        r"\bhow (?:do|can) i (?:fix|reset|cancel|return|connect|recover|unlock)\b",
        r"\bwhy (?:is|does|did|won't|wont|can't|cant) .*(?:work|working|connect|load|start)\b",
        r"\bmy .*(?:isn't|isnt|won't|wont|doesn't|doesnt|can't|cant)\b",
        r"\bi (?:need|want) (?:a )?(?:refund|replacement|return|cancel)\b",
    ]

    return any(re.search(pattern, lowered) for pattern in support_patterns)



def choose_route(query, support_results):
    best_score = support_results[0]["score"] if support_results else 0.0

    if has_support_cue(query) and best_score >= SUPPORT_ROUTE_THRESHOLD:
        return "support"

    return "general"


# ============================================================
# CONVERSATION CONTEXT
# ============================================================


def recent_transcript(messages, limit=8):
    useful = messages[-limit:]
    lines = []

    for message in useful:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")

    return "\n".join(lines)


# ============================================================
# OPEN-DOMAIN / GENERATIVE RESPONSES
# ============================================================


def call_openai(client, instructions, prompt, use_web=True):
    kwargs = {
        "model": get_openai_model(),
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": 900,
    }

    if use_web:
        kwargs["tools"] = [{"type": "web_search"}]

    response = client.responses.create(**kwargs)
    return response.output_text.strip()



def generate_general_answer(client, user_query, messages):
    instructions = (
        "You are an accurate open-domain assistant inside an NLP class project. "
        "Answer the user's actual question directly and clearly. Use web search when "
        "the answer depends on current, changing, niche, or externally verifiable facts. "
        "Do not fabricate facts, citations, links, personal access, or certainty. If the "
        "question is ambiguous, ask a concise clarifying question. If you are uncertain, "
        "say what is uncertain. Keep the answer useful rather than overly long."
    )

    transcript = recent_transcript(messages)

    prompt = (
        "Conversation context:\n"
        f"{transcript}\n\n"
        "Current user request:\n"
        f"{user_query}"
    )

    return call_openai(
        client,
        instructions,
        prompt,
        use_web=True,
    )



def generate_support_answer(client, user_query, support_results, messages):
    examples = []

    for number, result in enumerate(support_results, start=1):
        examples.append(
            f"Example {number}\n"
            f"Customer message: {result['message']}\n"
            f"Historical support response: {result['response']}\n"
            f"Retrieval score: {result['score']:.3f}"
        )

    retrieved_context = "\n\n".join(examples)
    transcript = recent_transcript(messages)

    instructions = (
        "You are a customer-support assistant in a hybrid NLP project. The project uses "
        "TF-IDF and MiniLM to retrieve historical support examples. Treat the retrieved "
        "examples as context, not as guaranteed truth. Answer the user's actual problem, "
        "not merely the closest historical message. Do not repeat Twitter handles, missing "
        "links, or phrases such as 'see this article' unless a real usable source is available. "
        "Use web search when current official product guidance would materially improve the "
        "answer. Prefer clear troubleshooting steps. Never claim that you accessed the user's "
        "account, device, order, or private information. If the retrieved examples are weak or "
        "conflicting, say what additional detail is needed rather than inventing a solution."
    )

    prompt = (
        "Conversation context:\n"
        f"{transcript}\n\n"
        "User's current support question:\n"
        f"{user_query}\n\n"
        "Retrieved customer-support examples from the TF-IDF + MiniLM model:\n"
        f"{retrieved_context}"
    )

    return call_openai(
        client,
        instructions,
        prompt,
        use_web=True,
    )


# ============================================================
# FALLBACK WHEN NO API KEY EXISTS
# ============================================================


def retrieval_only_fallback(query, support_results):
    if not support_results:
        return (
            "I cannot generate an open-domain answer because the OpenAI API key is not "
            "configured, and I did not find a support example for this question."
        )

    best = support_results[0]

    if has_support_cue(query) and best["score"] >= RAW_RETRIEVAL_THRESHOLD:
        return best["response"]

    return (
        "Open-domain answering is not enabled yet because this app does not have an "
        "OPENAI_API_KEY. The TF-IDF + MiniLM support retriever is available, but this "
        "question is outside its reliable support domain."
    )


# ============================================================
# MAIN HYBRID RESPONSE PIPELINE
# ============================================================


def answer_user_query(query, support_resources, client, messages):
    support_results = retrieve_support_examples(
        query,
        support_resources,
        top_k=TOP_K,
    )

    route = choose_route(query, support_results)

    if client is None:
        answer = retrieval_only_fallback(query, support_results)
        return answer, route, support_results

    if route == "support":
        answer = generate_support_answer(
            client,
            query,
            support_results,
            messages,
        )
    else:
        answer = generate_general_answer(
            client,
            query,
            messages,
        )

    return answer, route, support_results


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================


def run_chatbot():
    st.set_page_config(
        page_title="Hybrid Open-Domain NLP Chatbot",
        page_icon="💬",
        layout="centered",
    )

    st.title("Hybrid Open-Domain NLP Chatbot")

    st.caption(
        "TF-IDF + MiniLM customer-support retrieval combined with an "
        "open-domain language model and web-assisted answering."
    )

    if not artifacts_exist():
        st.warning("The TF-IDF + MiniLM support model has not been trained yet.")
        st.write("Place `twcs.csv` in this folder and run:")
        st.code("python chatbot.py --train")
        st.stop()

    try:
        support_resources = load_support_resources()
    except Exception as error:
        st.error(f"Unable to load the support model: {error}")
        st.stop()

    api_key = get_openai_api_key()
    client = get_openai_client(api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. You can ask me a general question or describe a "
                    "customer-service or technical-support problem."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("route"):
                with st.expander("Model details"):
                    st.write("Route:", message["route"])

                    if message.get("retrieval_score") is not None:
                        st.write(
                            "Best support retrieval score:",
                            f"{message['retrieval_score']:.3f}",
                        )

                    if message.get("matched_message"):
                        st.write("Closest support training message:")
                        st.write(message["matched_message"])

    user_input = st.chat_input("Ask any question...")

    if user_input:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, route, support_results = answer_user_query(
                        user_input,
                        support_resources,
                        client,
                        st.session_state.messages,
                    )
                except Exception as error:
                    answer = (
                        "I ran into an error while generating the answer. "
                        f"Technical detail: {error}"
                    )
                    route = "error"
                    support_results = []

            st.markdown(answer)

            best = support_results[0] if support_results else None

            with st.expander("Model details"):
                st.write("Route:", route)
                st.write("Open-domain model:", get_openai_model() if client else "Not configured")

                if best:
                    st.write("Best support retrieval score:", f"{best['score']:.3f}")
                    st.write("Semantic score:", f"{best['semantic_score']:.3f}")
                    st.write("TF-IDF score:", f"{best['lexical_score']:.3f}")
                    st.write("Closest support training message:")
                    st.write(best["message"])

        best = support_results[0] if support_results else None

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "route": route,
                "retrieval_score": best["score"] if best else None,
                "matched_message": best["message"] if best else None,
            }
        )

    with st.sidebar:
        st.header("Project Architecture")
        st.write("**Mode:** Hybrid open-domain")
        st.write("**Support NLP:** TF-IDF + MiniLM")
        st.write("**Support retrieval:** cosine similarity + entity re-ranking")
        st.write("**Generative model:**", get_openai_model())
        st.write("**General knowledge:** language model + web search")
        st.write("**Text augmentation:**", f"{AUGMENT_RATE:.0%}")
        st.write("**Top support candidates:**", TOP_K)

        if api_key:
            st.success("Open-domain API: configured")
        else:
            st.warning("Open-domain API: OPENAI_API_KEY not configured")

        st.divider()
        st.caption(
            "The support retriever is part of the project's trained NLP pipeline. "
            "The generative model is used for open-domain answering and to turn "
            "retrieved support examples into clearer responses."
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if "--train" in sys.argv:
        train_chatbot()
    else:
        run_chatbot()