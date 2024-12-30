import os
import faiss
import json
import torch
import tqdm
import numpy as np
import requests
import subprocess
import sys
import zipfile
import tarfile
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

##############################################################################
#                          自訂字典 (corpus_names / retriever_names)         #
##############################################################################
corpus_names = {
    "PubMed": ["pubmed"],
    "Textbooks": ["textbooks"],
    "StatPearls": ["statpearls"],
    "Wikipedia": ["wikipedia"],
    "train": ["train"],  # 若你有本地 train 資料夾，就這樣加回來
    "MedText": ["textbooks", "statpearls"],
    "MedCorp": ["pubmed", "textbooks", "statpearls", "wikipedia"],
}

retriever_names = {
    "BM25": ["bm25"],
    "Contriever": ["facebook/contriever"],
    "SPECTER": ["allenai/specter"],
    "MedCPT": ["ncbi/MedCPT-Query-Encoder"],
    "RRF-2": ["bm25", "ncbi/MedCPT-Query-Encoder"],
    "RRF-4": ["bm25", "facebook/contriever", "allenai/specter", "ncbi/MedCPT-Query-Encoder"]
}

##############################################################################
#                          工具函式 (下載、解壓)                             #
##############################################################################

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def download_file(url, dest_path):
    """使用 requests 來下載檔案"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def extract_tar(tar_path, extract_to):
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_to)

##############################################################################
#                     自訂 SentenceTransformer (CLS Pooling)                #
##############################################################################
class CustomizeSentenceTransformer(SentenceTransformer):
    """
    將原本預設的 MEAN pooling 改成 CLS pooling
    """
    def _load_auto_model(self, model_name_or_path, *args, **kwargs):
        print("No sentence-transformers model found with name {}. Creating a new one with CLS pooling.".format(model_name_or_path))
        token = kwargs.get('token', None)
        cache_folder = kwargs.get('cache_folder', None)
        revision = kwargs.get('revision', None)
        trust_remote_code = kwargs.get('trust_remote_code', False)

        if any(key in kwargs for key in ['token', 'cache_folder', 'revision', 'trust_remote_code']):
            transformer_model = Transformer(
                model_name_or_path,
                cache_dir=cache_folder,
                model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
                tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            )
        else:
            transformer_model = Transformer(model_name_or_path)

        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), 'cls')
        return [transformer_model, pooling_model]

##############################################################################
#                        embedding / construct_index                        #
##############################################################################
def embed(chunk_dir, index_dir, model_name, **kwarg):
    """
    將 chunk_dir 中的 .jsonl 檔案進行 embedding 並輸出成 .npy
    如果尚未有預先下載的 embedding，就在這裡 encode
    """
    save_dir = os.path.join(index_dir, "embedding")

    # 根據 model_name 選擇 Contriever or CustomizeSentenceTransformer
    if "contriever" in model_name:
        model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = CustomizeSentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    # 蒐集所有 .jsonl 檔案
    fnames = sorted([fname for fname in os.listdir(chunk_dir) if fname.endswith(".jsonl")])

    # 若 embedding 目錄不存在就建立
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for fname in tqdm.tqdm(fnames):
            fpath = os.path.join(chunk_dir, fname)
            save_path = os.path.join(save_dir, fname.replace(".jsonl", ".npy"))

            # 如果已經有對應的 .npy，略過
            if os.path.exists(save_path):
                continue

            # 以 UTF-8 讀取檔案，避免編碼錯誤
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read().strip()
            if not content:
                continue

            lines = content.split('\n')
            texts_data = [json.loads(item) for item in lines]

            # 依不同 model_name 決定如何處理文本
            if "specter" in model_name.lower():
                texts = [model.tokenizer.sep_token.join([item["title"], item["content"]]) for item in texts_data]
            elif "contriever" in model_name.lower():
                texts = [". ".join([item["title"], item["content"]]).replace('..', '.').replace("?.", "?") for item in texts_data]
            elif "medcpt" in model_name.lower():
                texts = [[item["title"], item["content"]] for item in texts_data]
            else:
                texts = [concat(item["title"], item["content"]) for item in texts_data]

            embed_chunks = model.encode(texts, **kwarg)
            np.save(save_path, embed_chunks)

        # encode 一個空字串，只是為了得到 embedding 維度
        embed_chunks = model.encode([""], **kwarg)

    return embed_chunks.shape[-1]

def construct_index(index_dir, model_name, h_dim=768, HNSW=False, M=32):
    """
    讀取 embedding/ 目錄下的 .npy 檔，寫入到 faiss 索引中
    並將 metadatas 記錄到 metadatas.jsonl
    """
    meta_path = os.path.join(index_dir, "metadatas.jsonl")
    # 先將 metadatas.jsonl 清空
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write("")

    # 決定使用 HNSW 或 Flat 索引
    if HNSW:
        if "specter" in model_name.lower():
            index = faiss.IndexHNSWFlat(h_dim, M)
        else:
            index = faiss.IndexHNSWFlat(h_dim, M)
            index.metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        if "specter" in model_name.lower():
            index = faiss.IndexFlatL2(h_dim)
        else:
            index = faiss.IndexFlatIP(h_dim)

    embed_dir = os.path.join(index_dir, "embedding")
    for fname in tqdm.tqdm(sorted(os.listdir(embed_dir))):
        if not fname.endswith(".npy"):
            continue
        arr_path = os.path.join(embed_dir, fname)
        curr_embed = np.load(arr_path)
        index.add(curr_embed)
        with open(meta_path, 'a', encoding='utf-8', errors='replace') as f:
            for i in range(len(curr_embed)):
                meta_dict = {'index': i, 'source': fname.replace(".npy", "")}
                f.write(json.dumps(meta_dict) + "\n")

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    return index

##############################################################################
#                                Retriever                                  #
##############################################################################
class Retriever:
    def __init__(self, retriever_name="ncbi/MedCPT-Query-Encoder", corpus_name="textbooks",
                 db_dir="./corpus", HNSW=False, **kwarg):

        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir

        # 確保基礎資料夾存在
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        # chunk_dir：存放 .jsonl 檔的目錄
        self.chunk_dir = os.path.join(self.db_dir, self.corpus_name, "chunk")

        # 若 chunk_dir 不存在，才嘗試 git clone
        # 如果你確定本地端已放好資料，可將整段 clone 註解或加邏輯判斷
        if not os.path.exists(self.chunk_dir):
            print(f"Chunk directory not found: {self.chunk_dir}")
            print(f"Skipping git clone for local usage: corpus_name = {self.corpus_name}")
            # 如果你真的需要從 Hugging Face clone，才保留以下幾行，否則可註解
            # print("Cloning the {:s} corpus from Huggingface...".format(self.corpus_name))
            # subprocess.run([
            #     "git", "clone",
            #     f"https://huggingface.co/datasets/MedRAG/{self.corpus_name}",
            #     os.path.join(self.db_dir, self.corpus_name)
            # ], check=True)
            #
            # if self.corpus_name == "statpearls":
            #     print("Downloading the statpearls corpus from NCBI bookshelf...")
            #     tar_path = os.path.join(self.db_dir, self.corpus_name, "statpearls_NBK430685.tar.gz")
            #     download_file("https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz", tar_path)
            #     extract_tar(tar_path, os.path.join(self.db_dir, self.corpus_name))
            #     print("Chunking the statpearls corpus...")
            #     subprocess.run([sys.executable, "src/data/statpearls.py"], check=True)

        # 建立 index_dir
        self.index_dir = os.path.join(self.db_dir, self.corpus_name, "index",
                                      self.retriever_name.replace("Query-Encoder", "Article-Encoder"))

        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            self.metadatas = None
            self.embedding_function = None
            if os.path.exists(self.index_dir):
                self.index = LuceneSearcher(os.path.join(self.index_dir))
            else:
                # 建立 BM25 index
                subprocess.run([
                    sys.executable, "-m", "pyserini.index.lucene",
                    "--collection", "JsonCollection",
                    "--input", self.chunk_dir,
                    "--index", self.index_dir,
                    "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "16"
                ], check=True)
                self.index = LuceneSearcher(os.path.join(self.index_dir))
        else:
            faiss_path = os.path.join(self.index_dir, "faiss.index")
            if os.path.exists(faiss_path):
                # 讀取已經存在的 faiss 索引
                self.index = faiss.read_index(faiss_path)
                meta_path = os.path.join(self.index_dir, "metadatas.jsonl")
                with open(meta_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.read().strip().split('\n')
                self.metadatas = [json.loads(line) for line in lines]
            else:
                # 若沒有 faiss.index，就嘗試嵌入後建立索引
                print(f"[In progress] Embedding the {self.corpus_name} corpus "
                      f"with the {self.retriever_name.replace('Query-Encoder', 'Article-Encoder')} retriever...")

                embed_dir = os.path.join(self.index_dir, "embedding")

                # 判斷有沒有預先下載 embedding
                if (self.corpus_name in ["textbooks", "pubmed", "wikipedia"]
                        and self.retriever_name in ["allenai/specter", "facebook/contriever", "ncbi/MedCPT-Query-Encoder"]
                        and not os.path.exists(embed_dir)):
                    print(f"[In progress] Downloading the {self.corpus_name} embeddings "
                          f"given by the {self.retriever_name.replace('Query-Encoder', 'Article-Encoder')} model...")
                    os.makedirs(self.index_dir, exist_ok=True)

                    # 下面的 URL 需要你自行更新為可下載的實際連結
                    embed_zip_path = os.path.join(self.index_dir, "embedding.zip")
                    url = "https://myuva-my.sharepoint.com/..."  # <--- 這裡放實際可下載的 URL
                    download_file(url, embed_zip_path)
                    extract_zip(embed_zip_path, self.index_dir)
                    os.remove(embed_zip_path)
                    h_dim = 768
                else:
                    # 直接做 embedding
                    h_dim = embed(
                        chunk_dir=self.chunk_dir,
                        index_dir=self.index_dir,
                        model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"),
                        **kwarg
                    )

                print(f"[In progress] Embedding finished! The dimension of the embeddings is {h_dim}.")
                self.index = construct_index(
                    index_dir=self.index_dir,
                    model_name=self.retriever_name.replace("Query-Encoder", "Article-Encoder"),
                    h_dim=h_dim,
                    HNSW=HNSW
                )
                print("[Finished] Corpus indexing finished!")

                meta_path = os.path.join(self.index_dir, "metadatas.jsonl")
                with open(meta_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.read().strip().split('\n')
                self.metadatas = [json.loads(line) for line in lines]

            # 建立 embedding_function
            if "contriever" in self.retriever_name.lower():
                self.embedding_function = SentenceTransformer(
                    self.retriever_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.embedding_function = CustomizeSentenceTransformer(
                    self.retriever_name,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            self.embedding_function.eval()

    def get_relevant_documents(self, question, k=32, id_only=False, **kwarg):
        """
        給定問題 question，回傳最相似的文件
        """
        assert isinstance(question, str)
        question = [question]

        # 若是 bm25
        if "bm25" in self.retriever_name.lower():
            from pyserini.search.lucene import LuceneSearcher
            res_ = [[]]
            hits = self.index.search(question[0], k=k)
            scores_array = np.array([h.score for h in hits])
            res_[0].append(scores_array)
            ids = [h.docid for h in hits]
            indices = [{
                "source": '_'.join(h.docid.split('_')[:-1]),
                "index": eval(h.docid.split('_')[-1])
            } for h in hits]
        else:
            with torch.no_grad():
                query_embed = self.embedding_function.encode(question, **kwarg)
            res_ = self.index.search(query_embed, k=k)
            # res_[0][0] = scores, res_[1][0] = index
            ids = [
                '_'.join([self.metadatas[i]["source"], str(self.metadatas[i]["index"])])
                for i in res_[1][0]
            ]
            indices = [self.metadatas[i] for i in res_[1][0]]

        scores = res_[0][0].tolist()

        if id_only:
            return [{"id": i} for i in ids], scores
        else:
            # 透過 idx2txt 取得實際內容
            return self.idx2txt(indices), scores

    def idx2txt(self, indices):
        """
        讀取 chunk 目錄下對應 .jsonl 檔案
        indices: [{"source": str, "index": int}, ...]
        """
        results = []
        for i in indices:
            jsonl_path = os.path.join(self.chunk_dir, i["source"] + ".jsonl")
            with open(jsonl_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.read().strip().split('\n')
            data = json.loads(lines[i["index"]])
            results.append(data)
        return results

##############################################################################
#                             RetrievalSystem                                #
##############################################################################
class RetrievalSystem:
    def __init__(self, retriever_name="MedCPT", corpus_name="Textbooks",
                 db_dir="./corpus", HNSW=False, cache=False):
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        assert self.corpus_name in corpus_names
        assert self.retriever_name in retriever_names

        self.retrievers = []
        for retriever in retriever_names[self.retriever_name]:
            row_list = []
            for corpus in corpus_names[self.corpus_name]:
                row_list.append(Retriever(retriever, corpus, db_dir, HNSW=HNSW))
            self.retrievers.append(row_list)

        self.cache = cache
        if self.cache:
            self.docExt = DocExtracter(cache=True, corpus_name=self.corpus_name, db_dir=db_dir)
        else:
            self.docExt = None

    def retrieve(self, question, k=32, rrf_k=100, id_only=False):
        """
        給定 question，回傳前 k 筆結果
        如果有 RRF (Reciprocal Rank Fusion)，則會合併多個 retriever 的結果
        """
        assert isinstance(question, str)

        if self.cache:
            # 若開啟 cache 模式，就先將 id_only 設為 True (只取 id)
            id_only = True

        texts = []
        scores = []

        # 若為 RRF
        if "RRF" in self.retriever_name:
            k_ = max(k * 2, 100)
        else:
            k_ = k

        for i in range(len(retriever_names[self.retriever_name])):
            texts.append([])
            scores.append([])
            for j in range(len(corpus_names[self.corpus_name])):
                t, s = self.retrievers[i][j].get_relevant_documents(question, k=k_, id_only=id_only)
                texts[-1].append(t)
                scores[-1].append(s)

        # 合併結果
        merged_texts, merged_scores = self.merge(texts, scores, k=k, rrf_k=rrf_k)

        # 若有開啟 cache 模式，將 id -> 實際文本
        if self.cache:
            merged_texts = self.docExt.extract(merged_texts)

        return merged_texts, merged_scores

    def merge(self, texts, scores, k=32, rrf_k=100):
        """
        若有多個 retriever，使用 RRF (Reciprocal Rank Fusion) 整合
        """
        RRF_dict = {}

        for i in range(len(retriever_names[self.retriever_name])):
            # 把同 retriever 不同 corpus 的結果合併
            texts_all = None
            scores_all = None
            for j in range(len(corpus_names[self.corpus_name])):
                if texts_all is None:
                    texts_all = texts[i][j]
                    scores_all = scores[i][j]
                else:
                    texts_all = texts_all + texts[i][j]
                    scores_all = scores_all + scores[i][j]

            # 依分數排序
            if "specter" in retriever_names[self.retriever_name][i].lower():
                # specter => L2距離 => 越小越好 => ascending
                sorted_index = np.array(scores_all).argsort()
            else:
                # 其餘 => 分數越大越好 => descending
                sorted_index = np.array(scores_all).argsort()[::-1]

            # 重新排序
            texts_sorted = [texts_all[idx] for idx in sorted_index]
            scores_sorted = [scores_all[idx] for idx in sorted_index]

            # 做 RRF
            for rank, item in enumerate(texts_sorted):
                _id = item["id"] if isinstance(item, dict) else item
                if _id in RRF_dict:
                    RRF_dict[_id]["score"] += 1 / (rrf_k + rank + 1)
                    RRF_dict[_id]["count"] += 1
                else:
                    RRF_dict[_id] = {
                        "id": _id,
                        "title": item.get("title", "") if isinstance(item, dict) else "",
                        "content": item.get("content", "") if isinstance(item, dict) else "",
                        "score": 1 / (rrf_k + rank + 1),
                        "count": 1
                    }

        RRF_list = sorted(RRF_dict.items(), key=lambda x: x[1]["score"], reverse=True)

        # 如果只有一個 retriever，不需做 RRF 合併
        if len(retriever_names[self.retriever_name]) == 1:
            # texts[0] 已是對應結果
            final_texts = texts[0][0][:k]
            final_scores = scores[0][0][:k]
        else:
            # 多個 retriever => 取前 k
            final_texts = []
            final_scores = []
            for item in RRF_list[:k]:
                data = {
                    "id": item[1]["id"],
                    "title": item[1]["title"],
                    "content": item[1]["content"]
                }
                final_texts.append(data)
                final_scores.append(item[1]["score"])

        return final_texts, final_scores

##############################################################################
#                              DocExtracter                                  #
##############################################################################
class DocExtracter:
    def __init__(self, db_dir="./corpus", cache=False, corpus_name="MedCorp"):
        self.db_dir = db_dir
        self.cache = cache
        print("Initializing the document extracter...")

        # 確認各個 corpus 都有 chunk；若無則嘗試 clone (可視需求改寫)
        for corpus in corpus_names[corpus_name]:
            chunk_path = os.path.join(self.db_dir, corpus, "chunk")
            if not os.path.exists(chunk_path):
                print("Local chunk path not found:", chunk_path)
                print(f"Skipping git clone for local usage: corpus = {corpus}")
                # 若需要 clone，可取消註解
                # subprocess.run(["git", "clone",
                #                f"https://huggingface.co/datasets/MedRAG/{corpus}",
                #                os.path.join(self.db_dir, corpus)],
                #               check=True)
                #
                # if corpus == "statpearls":
                #     print("Downloading the statpearls corpus from NCBI bookshelf...")
                #     tar_path = os.path.join(self.db_dir, corpus, "statpearls_NBK430685.tar.gz")
                #     download_file("https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz", tar_path)
                #     extract_tar(tar_path, os.path.join(self.db_dir, corpus))
                #     print("Chunking the statpearls corpus...")
                #     subprocess.run([sys.executable, "src/data/statpearls.py"], check=True)

        # 根據 cache 與否，初始化 dict
        if self.cache:
            json_path = os.path.join(self.db_dir, "_".join([corpus_name, "id2text.json"]))
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8', errors='replace') as f:
                    self.dict = json.load(f)
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    chunk_dir = os.path.join(self.db_dir, corpus, "chunk")
                    if not os.path.exists(chunk_dir):
                        continue
                    for fname in tqdm.tqdm(sorted(os.listdir(chunk_dir))):
                        fpath = os.path.join(chunk_dir, fname)
                        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read().strip()
                        if not content:
                            continue
                        for line in content.split('\n'):
                            item = json.loads(line)
                            # pop 不需要的欄位
                            item.pop("contents", None)
                            self.dict[item["id"]] = item
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.dict, f)
        else:
            path_json = os.path.join(self.db_dir, "_".join([corpus_name, "id2path.json"]))
            if os.path.exists(path_json):
                with open(path_json, 'r', encoding='utf-8', errors='replace') as f:
                    self.dict = json.load(f)
            else:
                self.dict = {}
                for corpus in corpus_names[corpus_name]:
                    chunk_dir = os.path.join(self.db_dir, corpus, "chunk")
                    if not os.path.exists(chunk_dir):
                        continue
                    for fname in tqdm.tqdm(sorted(os.listdir(chunk_dir))):
                        fpath = os.path.join(chunk_dir, fname)
                        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read().strip()
                        if not content:
                            continue
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            item = json.loads(line)
                            self.dict[item["id"]] = {"fpath": os.path.join(corpus, "chunk", fname), "index": i}

                with open(path_json, 'w', encoding='utf-8') as f:
                    json.dump(self.dict, f, indent=4)

        print("Initialization finished!")

    def extract(self, ids):
        """
        cache=True 時，self.dict[id] 直接存完整資訊
        cache=False 時，需要再讀一次 .jsonl 找到對應行
        """
        output = []
        if self.cache:
            for i in ids:
                _id = i if isinstance(i, str) else i["id"]
                output.append(self.dict[_id])
        else:
            for i in ids:
                _id = i if isinstance(i, str) else i["id"]
                info = self.dict[_id]
                real_fpath = os.path.join(self.db_dir, info["fpath"])
                with open(real_fpath, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read().strip().split('\n')
                data = json.loads(content[info["index"]])
                output.append(data)

        return output
