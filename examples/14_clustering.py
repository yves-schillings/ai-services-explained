"""Clustering and dimensionality reduction: group similar documents without labels.

Author: Yves Schillings.

Run:  python examples/14_clustering.py

Need: an analyst receives a large set of documents and wants to see which kinds
exist before anyone labels them. There is no answer column, so this is
unsupervised learning.

Steps: TF-IDF turns texts into thousands of numbers. TruncatedSVD (latent semantic
analysis) reduces them to a few dimensions that keep most of the variation.
K-means groups the documents. The silhouette score helps choose the number of
groups, and the analyst names each group from its top words.
The hidden labels of the fictional data are used only at the end, to show how
well the groups match them. Real data has no such check.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from fictional_data import CASE_WORDS, RESULTS, make_case_documents, save_json


def main():
    docs = make_case_documents(n_cases=200, seed=21)
    # Case place names are removed so that the groups reflect the kind of document.
    texts = [" ".join(w for w in d["text"].split() if w not in CASE_WORDS) for d in docs]
    tfidf = TfidfVectorizer(min_df=3)
    x = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=20, random_state=0)
    z = svd.fit_transform(x)
    z = z / np.linalg.norm(z, axis=1, keepdims=True)
    print(f"TF-IDF: {x.shape[1]} dimensions. SVD keeps 20, explaining "
          f"{svd.explained_variance_ratio_.sum():.0%} of the variance.")
    sil = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(z)
        sil.append({"k": k, "silhouette": round(float(silhouette_score(z, km.labels_)), 3)})
    best_k = max(sil, key=lambda r: r["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit(z)
    terms = np.array(tfidf.get_feature_names_out())
    centroids = svd.inverse_transform(km.cluster_centers_)
    top = {int(c): list(terms[np.argsort(-centroids[c])[:6]]) for c in range(best_k)}
    ari = adjusted_rand_score([d["label"] for d in docs], km.labels_)
    print("Silhouette by k:", sil, "-> k =", best_k)
    print("Top words per group (the analyst names each group):", top)
    print(f"Agreement with the hidden labels (adjusted Rand index): {ari:.2f}")
    print("Limit: a group is a proposal. The analyst checks samples before naming or using it.")
    result = {"documents": len(docs), "tfidf_dimensions": int(x.shape[1]), "svd_components": 20,
              "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 3), "silhouette": sil,
              "chosen_k": best_k, "top_words": top, "adjusted_rand_index": round(float(ari), 3)}
    save_json("14_clustering.json", result)
    plot(z, km.labels_, sil)
    return result


def plot(z, labels, sil):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([r["k"] for r in sil], [r["silhouette"] for r in sil], "o-")
    ax[0].set_xlabel("number of groups k")
    ax[0].set_ylabel("silhouette score")
    ax[0].set_title("Choose k: higher silhouette means clearer groups")
    ax[1].scatter(z[:, 0], z[:, 1], c=labels, s=8, cmap="tab10")
    ax[1].set_xlabel("SVD component 1")
    ax[1].set_ylabel("SVD component 2")
    ax[1].set_title("Documents in two SVD dimensions, coloured by group")
    fig.tight_layout()
    RESULTS.mkdir(exist_ok=True)
    fig.savefig(RESULTS / "14_clustering.png", dpi=130)


if __name__ == "__main__":
    main()
