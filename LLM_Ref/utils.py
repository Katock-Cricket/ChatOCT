import json
import pickle as pkl

import numpy as np
from text2vec import SentenceModel

msd_dict = json.load(open('LLM_Ref/msd_dict.json', 'r', encoding='utf-8'))
disease_dict = json.load(open('LLM_Ref/disease_dict.json', 'r', encoding='utf-8'))


def get_ref(query: str):
    topic_list = query_range(query, k=1, bar=0.0)
    if len(topic_list) == 0:
        return None
    info_dict = info_list(topic_list)
    print(f"查询到info：\n{info_dict}\n查询到topics：\n{topic_list}\n")
    try:
        url = msd_dict[topic_list[0]]
    except:
        url = "www.msdmanuals.cn/professional"
    return info_dict, "https://" + url


def info_list(topic_list):
    info_base = {i: disease_dict[i] for i in topic_list}
    return info_base


def query_range(query: str, k: int = 3, bar=0.6):
    emb_d = pkl.load(open('LLM_Ref/MSD.pkl', 'rb'))
    embeddings = []
    for key, value in emb_d.items():
        embeddings.append(value)
    embeddings = np.asarray(embeddings)
    model = SentenceModel()
    q_emb = model.encode(query)
    # q_emb = m.encode(query)
    q_emb = q_emb / np.linalg.norm(q_emb, ord=2)

    # Calculate the cosine similarity between the query embedding and all other embeddings
    cos_similarities = np.dot(embeddings, q_emb)

    # Get the indices of the embeddings with the highest cosine similarity scores
    top_k_indices = cos_similarities.argsort()[-k:][::-1]
    print(cos_similarities[top_k_indices])
    sift_topK = top_k_indices[np.argwhere(cos_similarities[top_k_indices] > bar)]
    sift_topK = sift_topK.reshape(sift_topK.shape[0], )
    ret = []
    if len(sift_topK) == 0:
        return ret
    # for indices in top_k_indices:
    for indices in sift_topK:
        key = list(emb_d.keys())[indices]
        ret.append(key)
        print(key)
        # print(msd[key])
    return ret
