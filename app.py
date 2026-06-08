from flask import Flask, render_template, request
import pandas as pd
import re
import os
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# ==================================
# LOAD DATA
# ==================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

paper_path = os.path.join(
    BASE_DIR,
    "hasil_scraping_okezone.xlsx"
)

pre_path = os.path.join(
    BASE_DIR,
    "hasil_preprocessing.xlsx"
)

paper_x = pd.read_excel(paper_path)
paper = paper_x.values.tolist()

df_pre = pd.read_excel(pre_path)

processed_paper = (
    df_pre["final_text"]
    .fillna("")
    .astype(str)
    .tolist()
)

# ==================================
# NLP SETUP
# ==================================

stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True
)

tfidf_matrix = vectorizer.fit_transform(
    processed_paper
)

# ==================================
# HIGHLIGHT
# ==================================

def highlight(text, keyword):

    for word in keyword.split():

        pattern = rf"\b({re.escape(word)})\b"

        text = re.sub(
            pattern,
            r"<mark>\1</mark>",
            text,
            flags=re.IGNORECASE
        )

    return text

# ==================================
# SEARCH
# ==================================

@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    query_asli = ""

    if request.method == "POST":

        query_asli = request.form.get(
            "query",
            ""
        ).strip()

        query = query_asli.lower()

        query = query.translate(
            str.maketrans(
                '',
                '',
                string.punctuation
            )
        )

        query = stopword.remove(query)

        tokens = query.split()

        if len(tokens) > 0:

            tokens_stem = [
                stemmer.stem(t)
                for t in tokens
            ]

            query_joined = " ".join(
                tokens_stem
            )

            query_vec = vectorizer.transform(
                [query_joined]
            )

            similarity = cosine_similarity(
                tfidf_matrix,
                query_vec
            ).flatten()

            ranked_idx = np.argsort(
                -similarity
            )

            for idx in ranked_idx:

                score = float(
                    similarity[idx]
                )

                # buang similarity kecil
                if score < 0.01:
                    continue

                judul = str(
                    paper[idx][0]
                )

                tanggal = str(
                    paper[idx][1]
                )

                isi = str(
                    paper[idx][2]
                )

                link = str(
                    paper[idx][3]
                )

                teks_gabungan = (
                    judul.lower()
                    + " "
                    + isi.lower()
                )

                matched = 0

                for token in tokens:

                    if re.search(
                        rf"\b{re.escape(token)}\b",
                        teks_gabungan,
                        re.IGNORECASE
                    ):
                        matched += 1

                # WAJIB cocok mayoritas keyword
                minimal_match = max(
                    1,
                    int(
                        np.ceil(
                            len(tokens) * 0.6
                        )
                    )
                )

                if matched < minimal_match:
                    continue

                # snippet
                match = re.search(
                    rf"\b{re.escape(tokens[0])}\b",
                    isi,
                    re.IGNORECASE
                )

                if match:

                    pos = match.start()

                    start = max(
                        0,
                        pos - 60
                    )

                    snippet = isi[
                        start:
                        start + 220
                    ]

                else:

                    snippet = isi[:220]

                snippet += "..."

                judul = highlight(
                    judul,
                    query_asli
                )

                snippet = highlight(
                    snippet,
                    query_asli
                )

                results.append({
                    "judul": judul,
                    "tanggal": tanggal,
                    "snippet": snippet,
                    "link": link,
                    "score": round(
                        score,
                        4
                    )
                })

                if len(results) >= 50:
                    break

    return render_template(
        "index.html",
        results=results,
        query=query_asli
    )

# ==================================
# VERCEL
# ==================================

application = app

if __name__ == "__main__":
    app.run(debug=True)
