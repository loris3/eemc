import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
class BaseRepresentation(ABC):
    """
    Base class for document representations with per-document **disk caching**.
    """
    def __init__(self, name: str, cache_dir: str = "./cache/representations"):
        self.name = name
        self._cache = None  # full similarity matrix
        self.cache_dir = Path(cache_dir) / self.name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _compute_doc(self, idx: int) -> torch.Tensor:
        """
        Compute embedding for a single document by index.
        """
        pass

    def _doc_cache_path(self, idx: int) -> Path:
        """
        Return the path where this document's embedding will be cached.
        """
        return self.cache_dir / f"doc_{idx}.pt"

    def compute_doc(self, idx: int, force: bool = False) -> torch.Tensor:
        """
        Compute or load cached document embedding from disk.
        """
        path = self._doc_cache_path(idx)
        if path.exists() and not force:
            try:
                emb = torch.load(path, map_location="cpu")
                return emb
            except Exception as e:
                print(f"Failed loading cached embedding for doc {idx}: {e}, recomputing...")

        emb = self._compute_doc(idx)
        torch.save(emb, path)
        return emb

    def _compute(self) -> torch.Tensor:
        """
        Compute full similarity matrix using cached document embeddings.
        """
        embeddings = []
        n = self.num_documents()
        for i in tqdm(range(n), desc=f"Computing {self.name} embeddings"):
            embeddings.append(self.compute_doc(i))
        embeddings = torch.vstack(embeddings)
        sim_matrix = embeddings @ embeddings.T
        return self.normalize(sim_matrix)

    def compute(self, force: bool = False) -> torch.Tensor:
        if self._cache is None or force:
            self._cache = self._compute()
        return self._cache

    @abstractmethod
    def num_documents(self) -> int:
        pass

    @staticmethod
    def normalize(sim_matrix: torch.Tensor) -> torch.Tensor:
        return F.normalize(sim_matrix, dim=-1)

# ----------------- Gradient -----------------
class GradientRepresentation(BaseRepresentation):
    def __init__(self, estimator, dataset_name, dataset_split, indices):
        super().__init__(name=estimator.__class__.__name__)
        self.estimator = estimator
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.indices = indices

    def _compute_doc(self, idx: int) -> torch.Tensor:
        gradient = self.estimator.get_gradient(
            self.dataset_name,
            self.dataset_split,
            [self.indices[idx]]
        )[0].float()
        gradient = F.normalize(gradient.unsqueeze(0), p=2, dim=1)
        return gradient.squeeze(0).cpu()

    def num_documents(self):
        return len(self.indices)


# ----------------- Full Model -----------------
class FullModelRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, name="full_model"):
        super().__init__(name=name)
        self.model = model
        self.tokenized_dataset = tokenized_dataset

    def _compute_doc(self, idx: int) -> torch.Tensor:
        m = self.tokenized_dataset[idx]
        device = "cuda"
        self.model.to(device)
        self.model.eval()

        input_ids = torch.tensor([m["input_ids"]], device=device)
        attention_mask = torch.tensor([m["attention_mask"]], device=device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        token_embeddings = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        sentence_embedding = (masked_embeddings.sum(dim=1) / mask.sum(dim=1))
        sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
        return sentence_embedding.squeeze(0).cpu()

    def num_documents(self):
        return len(self.tokenized_dataset)


# ----------------- SBERT -----------------
class SBERTRepresentation(BaseRepresentation):
    def __init__(self, st_model, decoded_texts):
        super().__init__(name="sbert")
        self.st_model = st_model
        self.decoded_texts = decoded_texts

    def _compute_doc(self, idx: int) -> torch.Tensor:
        device = "cuda"
        self.st_model.to(device)
        text = self.decoded_texts[idx]
        emb = self.st_model.encode([text])[0]
        return F.normalize(torch.tensor(emb, dtype=torch.float32).unsqueeze(0), p=2, dim=1).squeeze(0).cpu()

    def num_documents(self):
        return len(self.decoded_texts)


# ----------------- BM25 -----------------
class BM25Representation(BaseRepresentation):
    def __init__(self, tokenized_docs):
        super().__init__(name="bm25")
        self.tokenized_docs = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs)

    def _compute_doc(self, idx: int) -> torch.Tensor:
        scores = self.bm25.get_scores(self.tokenized_docs[idx])
        return F.normalize(torch.tensor(scores, dtype=torch.float32).unsqueeze(0), p=2, dim=1).squeeze(0).cpu()

    def num_documents(self):
        return len(self.tokenized_docs)


# ----------------- TFIDF -----------------
class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs):
        super().__init__(name="tfidf")
        self.tokenized_docs = tokenized_docs
        self.docs_as_strings = [" ".join(map(str, doc)) for doc in tokenized_docs]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs_as_strings)

    def _compute_doc(self, idx: int) -> torch.Tensor:
        vec = torch.tensor(self.tfidf_matrix[idx].toarray(), dtype=torch.float32)
        return F.normalize(vec, p=2, dim=1).squeeze(0).cpu()

    def num_documents(self):
        return len(self.tokenized_docs)


# ----------------- Jaccard -----------------
class JaccardRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs):
        super().__init__(name="jaccard")
        self.tokenized_docs = tokenized_docs

    def _compute_doc(self, idx: int) -> torch.Tensor:
        set_i = set(self.tokenized_docs[idx])
        n = len(self.tokenized_docs)
        vec = torch.zeros(n, dtype=torch.float32)
        for j in range(n):
            set_j = set(self.tokenized_docs[j])
            union = set_i | set_j
            vec[j] = 0.0 if len(union) == 0 else len(set_i & set_j) / len(union)
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0).cpu()

    def num_documents(self):
        return len(self.tokenized_docs)


# ----------------- Overlap -----------------
class OverlapRepresentation(BaseRepresentation):
    def __init__(self, tokenized_docs):
        super().__init__(name="overlap")
        self.tokenized_docs = tokenized_docs

    def _compute_doc(self, idx: int) -> torch.Tensor:
        set_i = set(self.tokenized_docs[idx])
        n = len(self.tokenized_docs)
        vec = torch.zeros(n, dtype=torch.float32)
        for j in range(n):
            set_j = set(self.tokenized_docs[j])
            denom = min(len(set_i), len(set_j))
            vec[j] = 0.0 if denom == 0 else len(set_i & set_j) / denom
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0).cpu()

    def num_documents(self):
        return len(self.tokenized_docs)


# ----------------- Normalized Attention Map -----------------
import torch
import torch.nn.functional as F
from tqdm import tqdm

class NormalizedAttentionMapRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, max_seq_len=512, top_k_eig=128, name="spectral_attention"):
        super().__init__(name=name)
        self.model = model
        self.tokenized_dataset = tokenized_dataset
        self.max_seq_len = max_seq_len
        self.top_k_eig = top_k_eig

    def _compute_doc(self, idx: int) -> torch.Tensor:
        m = self.tokenized_dataset[idx]
        device = next(self.model.parameters()).device
        self.model.to(device)
        self.model.eval()

        input_ids = torch.tensor([m["input_ids"]], device=device)
        attention_mask = torch.tensor([m["attention_mask"]], device=device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        if outputs.attentions is None:
            raise ValueError("Model did not return attentions")

        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        transformer = getattr(base_model, "model", base_model)
        layers = transformer.layers


        value_acts = []
        dense_weights = []
        num_layers = len(outputs.attentions)

        for l in tqdm(range(num_layers), desc="2"):
            layer = layers[l]
            layer_hidden = outputs.hidden_states[l + 1] 


            v = layer.self_attn.v_proj(layer_hidden)  
            batch_size, seq_len_layer, hidden_size_layer = v.size()
            num_heads_layer = outputs.attentions[l].size(1)
            head_dim = hidden_size_layer // num_heads_layer

     
            v = v.view(batch_size, seq_len_layer, num_heads_layer, head_dim).permute(0, 2, 1, 3)
            value_acts.append(v)

     
            o_proj = layer.self_attn.o_proj
            w_orig = o_proj.weight  
            hidden_out, hidden_in = w_orig.size()
            w = w_orig.view(num_heads_layer, hidden_out // num_heads_layer, hidden_in)
            dense_weights.append(w)

 
                
        norm_maps = []

        for l in tqdm(range(num_layers), desc="3 (efficient)"):
            v = value_acts[l][0] 
            w = dense_weights[l]  

            alpha = outputs.attentions[l][0] 
            seq_len_use = min(alpha.size(1), layers[l].self_attn.o_proj.weight.size(1))

   
            layer_hidden = outputs.hidden_states[l + 1][0, :seq_len_use]  
            f_x = layers[l].self_attn.o_proj(layer_hidden) 


            for h in range(alpha.size(0)):  
                alpha_h = alpha[h, :seq_len_use, :seq_len_use]          
                f_x_norm = f_x.norm(dim=-1)                             
                norm_map = alpha_h * f_x_norm.unsqueeze(0)   


                if seq_len_use < self.max_seq_len:
                    pad = self.max_seq_len - seq_len_use
                    norm_map = F.pad(norm_map, (0, pad, 0, pad), value=0)

                norm_maps.append(norm_map)

        norm_maps = torch.stack(norm_maps)


        eigvals_all = torch.linalg.eigh(norm_maps, UPLO='U')[0] 
        topk_eigvals = eigvals_all[:, -self.top_k_eig:]  # top k per map
        doc_vector = topk_eigvals.mean(dim=0)  # average across all maps
        doc_vector = F.normalize(doc_vector.unsqueeze(0), p=2, dim=1).squeeze(0)
        return doc_vector.cpu()

    def num_documents(self):
        return len(self.tokenized_dataset)
# ----------------- RepSim emulating https://github.com/plumprc/Failures-of-Influence-Functions-in-LLMs/blob/93f9489872c0ee33d05c7b7316f48f6256e47a86/repsim.py -----------------
class RepSimRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, name="rep_sim"):
        super().__init__(name=name)
        self.model = model
        self.tokenized_dataset = tokenized_dataset

    def _compute_doc(self, idx: int) -> torch.Tensor:
        m = self.tokenized_dataset[idx]

        device = "cuda"
        self.model.to(device)
        self.model.eval()

        input_ids = torch.tensor([m["input_ids"]], device=device)
        attention_mask = torch.tensor([m["attention_mask"]], device=device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Last layer hidden states
        last_hidden = outputs.hidden_states[-1]   # (1, seq_len, hidden_dim)

        # Get last non-padding token
        seq_len = attention_mask.sum(dim=1) - 1
        last_token_embedding = last_hidden[0, seq_len, :]

        # L2 normalize (so dot product = cosine similarity)
        last_token_embedding = F.normalize(
            last_token_embedding.unsqueeze(0),
            p=2,
            dim=1
        ).squeeze(0)

        return last_token_embedding.cpu()

    def num_documents(self):
        return len(self.tokenized_dataset)