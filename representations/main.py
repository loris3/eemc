import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pickle

DEBUG = int(os.getenv("DEBUG", "0"))

if DEBUG:
    print(f"Debug mode enabled (DEBUG={DEBUG})")


def copy_file(src_file, src_root, dst_root):
    rel_path = os.path.relpath(src_file, src_root)
    dst_file = os.path.join(dst_root, rel_path)
    if os.path.exists(dst_file):
        return
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    try:
        subprocess.run(["cp", "-p", src_file, dst_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error copying {src_file} to {dst_file}: {e}")
        
        
# --- BaseRepresentation ---
class BaseRepresentation(ABC):
    def __init__(self, name, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split,
                 model_path, cache_dir: str = "./cache/representations", results_only=False, normalize_vectors=False):
        self.name = name
        self.train_dataset = train_dataset
        self.train_dataset_name = train_dataset_name
        self.train_dataset_split = train_dataset_split
        self.test_dataset = test_dataset
        self.test_dataset_name = test_dataset_name
        self.test_dataset_split = test_dataset_split
        self.model_path = model_path
        self.cache_dir = os.path.join(cache_dir, self.name, train_dataset_name, train_dataset_split,
                                      test_dataset_name, test_dataset_split, model_path)
        self.local_cache_dir = os.path.join("/tmp", cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.local_cache_dir, exist_ok=True)
        self.results_only = results_only
        self.normalize_vectors = normalize_vectors
        if self.results_only:
            self.copy_cache_to_tmp()
           
        
    def copy_cache_to_tmp(self):
            try:
                print("Copying cached representations", flush=True)
                folders = [
                    (self.cache_dir, self.local_cache_dir),
                ]

                max_workers = 32

                for remote_folder, local_folder in folders:
                    print(f"{remote_folder} --> {local_folder}", flush=True)
                    os.makedirs(remote_folder, exist_ok=True)
                    os.makedirs(local_folder, exist_ok=True)

                    all_files = []
                    for root, _, files in os.walk(remote_folder):
                        for f in files:
                            all_files.append(os.path.join(root, f))

                    # Split files evenly across workers
                    def chunks(lst, n):
                        k, m = divmod(len(lst), n)
                        return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

                    file_batches = chunks(all_files, max_workers)
                    def worker_copy(batch):
                        for f in batch:
                            copy_file(f, remote_folder, local_folder)

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(worker_copy, batch) for batch in file_batches]

           
                        with tqdm(total=len(futures), desc="Copying files") as pbar:
                            for future in as_completed(futures):
                                future.result()  
                                pbar.update(1)

            except Exception as ee:
                print(ee, flush=True)
                raise

    @abstractmethod
    def _embed_doc(self, idx: int) -> torch.Tensor:
        """
        Returns embedding for a single document.
        """
        pass

   

    def compute_doc(self, idx: int, force: bool = False) -> torch.Tensor:
        path = os.path.join(self.cache_dir, f"doc_{idx}.pt")
        if os.path.exists(path):
            try:
                return torch.load(path, map_location="cpu")
            except:
                pass
        if self.results_only:
            raise RuntimeError(f"results_only=True but cache not found for doc {idx}: {path}")
        emb = self._embed_doc(idx)
        torch.save(emb, path)
        return emb
    def compute(self, cache_embds=True) -> torch.Tensor:
        indices = self.test_dataset["indices"][0:DEBUG] if DEBUG else self.test_dataset["indices"]

        embeddings = [None] * len(indices) if self.results_only else None

        with tqdm(total=len(indices), desc=f"Computing {self.name} embeddings") as pbar:
            for pos, i in enumerate(indices):
                emb = self.compute_doc(i, force=not cache_embds)
                if self.results_only:
                    embeddings[pos] = emb
                pbar.update(1)

        if self.results_only:
            embeddings = torch.vstack(embeddings)
            if self.normalize_vectors:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
    

# --- GradientRepresentation ---
class GradientRepresentation(BaseRepresentation):
    def __init__(self, estimator, name="GradientRepresentation", results_only=False):
        super().__init__(name,
                         estimator.train_dataset,
                         estimator.train_dataset_name,
                         estimator.train_dataset_split,
                         estimator.test_dataset,
                         estimator.test_dataset_name,
                         estimator.test_dataset_split,
                         estimator.model_path,
                         results_only=results_only)
        self.estimator = estimator
        
    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")
        gradient = self.estimator.get_gradient(
            self.estimator.test_dataset_name,
            self.test_dataset_split,
            [idx]
        )[0].float()
       
        return gradient.cpu()

# --- FullModelRepresentation ---
class FullModelRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, name="full_model", results_only=False):

        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)

        self.tokenized_dataset = tokenized_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def _embed_doc(self, idx: int) -> torch.Tensor:

        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")

        m = self.tokenized_dataset[idx]

        input_ids = torch.tensor(m["input_ids"], device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(m["attention_mask"], device=self.device).unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        token_embeddings = outputs.hidden_states[-1]  # (1, n, d)
        mask = attention_mask.unsqueeze(-1).float()   # (1, n, 1)
        masked_embeddings = token_embeddings * mask

        sentence_embedding = masked_embeddings.sum(dim=1) / mask.sum(dim=1)

        return sentence_embedding.squeeze(0).cpu()


class SentenceTransformerRepresentation(BaseRepresentation):
    def __init__(self, model_path, decoded_texts, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 results_only=False):

        super().__init__(f"embd_{model_path}", train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)

        self.decoded_texts = decoded_texts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not results_only:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_path, trust_remote_code=True)
            self.st_model.to(self.device)

    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self.results_only:
            raise RuntimeError("results_only=True but no cache found")

        text = self.decoded_texts[idx]["text"]
        emb = self.st_model.encode([text])[0]
        return torch.tensor(emb, dtype=torch.float32).to("cpu")
    
# ----------------- NormalizedAttentionMapRepresentation -----------------
class NormalizedAttentionMapRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, top_k_eig=128, name="spectral_attention", results_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)
        self.model = model
        self.tokenized_dataset = tokenized_dataset
        self.max_seq_len = model.config.max_position_embeddings
        print("Max sequence length:", self.max_seq_len)
        self.top_k_eig = top_k_eig
        self.device ="cuda"
        self.model.to(self.device).eval()

    def _embed_doc(self, idx: int) -> torch.Tensor:
        m = self.tokenized_dataset[idx]
        input_ids = torch.tensor([m["input_ids"]], device=self.device)
        attention_mask = torch.tensor([m["attention_mask"]], device=self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        seq_len_use = min(input_ids.size(1), self.max_seq_len)
        num_layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].size(1)
        hidden_size = outputs.hidden_states[0].size(-1)
        head_size = hidden_size // num_heads

        base_model = getattr(self.model, "base_model", self.model)
        layers = base_model.model.model.layers

        norm_maps = []
        for l, alpha_layer in enumerate(outputs.attentions):
            layer_hidden = outputs.hidden_states[l + 1][0, :seq_len_use, :]  # (seq_len, hidden)
            dense_weight = layers[l].self_attn.o_proj.weight
            f_x = layer_hidden @ dense_weight.T
            f_x = f_x.view(seq_len_use, num_heads, head_size).permute(1, 0, 2)  # (heads, seq_len, head_size)
            f_x_norm = torch.linalg.norm(f_x, dim=-1)                             # (heads, seq_len)


            alpha_h_all = alpha_layer[0, :, :seq_len_use, :seq_len_use]           # (heads, seq_len, seq_len)
            f_x_norm_exp = f_x_norm.unsqueeze(2)                                  # (heads, seq_len, 1)
            layer_map = (alpha_h_all * f_x_norm_exp).mean(dim=0)                  # (seq_len, seq_len)

            if seq_len_use < self.max_seq_len:
                pad = self.max_seq_len - seq_len_use
                layer_map = F.pad(layer_map, (0, pad, 0, pad), value=0)

            norm_maps.append(layer_map)

        M_stacked = torch.stack(norm_maps, dim=0).reshape(-1, self.max_seq_len)
        u, s, v = torch.linalg.svd(M_stacked, full_matrices=False)
        k = min(self.top_k_eig, s.size(0))
        doc_vector = s[:k]
        print("doc_vector", doc_vector)
        return doc_vector.cpu()

# ----------------- RepSimRepresentation -----------------
class RepSimRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, name="rep_sim", results_only=False):

        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)

        self.tokenized_dataset = tokenized_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def _embed_doc(self, idx: int) -> torch.Tensor:

        if self.results_only:
            raise RuntimeError(f"results_only=True but cache not found for doc {idx}")

        m = self.tokenized_dataset[idx]

        input_ids = torch.tensor(m["input_ids"], device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(m["attention_mask"], device=self.device).unsqueeze(0)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        last_hidden = outputs.hidden_states[-1]

        # index of last non-padding token
        last_idx = attention_mask[0].nonzero()[-1].item()

        z = last_hidden[0, last_idx]   # shape: (d,)

        return z.cpu()

# ----------------- DataInfRepresentation -----------------
class DataInfRepresentation(BaseRepresentation):
    def __init__(self, estimator, results_only=False):
        super().__init__("datainf",
                         estimator.train_dataset,
                         estimator.train_dataset_name,
                         estimator.train_dataset_split,
                         estimator.test_dataset,
                         estimator.test_dataset_name,
                         estimator.test_dataset_split,
                         estimator.model_path,
                         results_only=results_only)
        self.estimator = estimator
        self._hvp_dict = None
        
    @property
    def hvp_dict(self):
        if self._hvp_dict is None:
            self.compute_hvp_approx_with_cache()
        return self._hvp_dict

    def compute_hvp_approx_with_cache(self, lambda_const_param=10, batch_size=32,
                                      val_batch_size=1000, inner_batch_size=1):
        estimator = self.estimator
        hvp_cache_path = os.path.join(estimator.gradient_out_dir, "hvp",
                                      estimator.train_dataset_name, estimator.train_dataset_split,
                                      estimator.test_dataset_name, estimator.test_dataset_split, "hvp.pt")
        os.makedirs(os.path.dirname(hvp_cache_path), exist_ok=True)

        if os.path.exists(hvp_cache_path):
            print(f"Loading cached HVP from {hvp_cache_path}")
            self._hvp_dict = torch.load(
                local_path,
                map_location="cpu",
                weights_only=False
                )
            print(f"Got cached HVP from {hvp_cache_path}")
            return

        if self.results_only:
            raise RuntimeError(f"results_only=True but HVP cache not found at {hvp_cache_path}")

      

    def _embed_doc(self, idx: int) -> torch.Tensor:
        if self._hvp_dict is None and self.results_only:
            raise RuntimeError(f"results_only=True but HVP cache not loaded for doc {idx}")
        gradient = torch.cat([self.hvp_dict[idx][k].flatten() for k in sorted(self.hvp_dict[idx].keys())])
        return gradient.cpu().float()
# ----------------- BM25 -----------------
class BM25Representation(BaseRepresentation):
    def __init__(self, tokenized_docs, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, name="bm25", results_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         "-", results_only=results_only)
        self.tokenized_docs = tokenized_docs if not results_only else None
        self.bm25 = BM25Okapi(tokenized_docs) if not results_only else None
        
    def _embed_doc(self, idx: int) -> torch.Tensor:
        scores = self.bm25.get_scores(self.tokenized_docs[idx])
        return torch.tensor(scores, dtype=torch.float32).cpu()


# ----------------- TFIDF -----------------
import torch
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.preprocessing import normalize
from representations.main import BaseRepresentation

class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, name="tfidf", results_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         "-", results_only=results_only)

        if not results_only:
            self.tokenized_docs = tokenized_docs
            # Convert tokenized docs to strings for CountVectorizer's "pretokenized" mode
            self.tokenized_str_docs = [[str(tok) for tok in doc] for doc in tokenized_docs]

            # Build count matrix
            self.count_vectorizer = CountVectorizer(
                tokenizer=lambda x: x,  # no-op tokenizer
                preprocessor=lambda x: x,  # no-op preprocessor
                lowercase=False
            )
            count_matrix = self.count_vectorizer.fit_transform(self.tokenized_str_docs)

            # Convert counts to TF-IDF
            tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
            self.tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

    def _embed_doc(self, idx: int) -> torch.Tensor:
        vec = torch.tensor(self.tfidf_matrix[idx].toarray(), dtype=torch.float32)
        return vec.squeeze(0).cpu()

# ----------------- Jaccard -----------------
class JaccardRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split, name="jaccard", results_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         "-", results_only=results_only)

        self.token_sets = [set(doc) for doc in tokenized_docs]

    def _embed_doc(self, idx: int) -> torch.Tensor:
        set_i = self.token_sets[idx]

        sims = []
        len_i = len(set_i)

        for set_j in self.token_sets:
            inter = len(set_i & set_j)
            union = len_i + len(set_j) - inter
            sims.append(0.0 if union == 0 else inter / union)

        return torch.tensor(sims, dtype=torch.float32)