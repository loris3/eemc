import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pathlib import Path
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from abc import ABC, abstractmethod
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DEBUG = True
class BaseRepresentation(ABC):
    def __init__(self, name, train_dataset, train_dataset_name, train_dataset_split, test_dataset, test_dataset_name, test_dataset_split, model_path, cache_dir: str = "./cache/representations"):
        self.name = name
        self.train_dataset = train_dataset
        self.train_dataset_name = train_dataset_name
        self.train_dataset_split = train_dataset_split
        self.test_dataset = test_dataset
        self.test_dataset_name = test_dataset_name
        self.test_dataset_split = test_dataset_split
        self.model_path = model_path
        self.cache_dir = os.path.join(cache_dir, self.name, train_dataset_name, train_dataset_split, test_dataset_name, test_dataset_split, model_path)
        os.makedirs(self.cache_dir, exist_ok=True)

    @abstractmethod
    def _embed_doc(self, idx: int) -> torch.Tensor:
        """
        Returns embedding for a single document.
        """
        pass

    def compute_doc(self, idx: int, force: bool = False) -> torch.Tensor:
        path = os.path.join(self.cache_dir, f"doc_{idx}.pt")
        if not force:
            try:
                emb = torch.load(path, map_location="cpu")
                return emb
            except Exception as e:
                pass
        emb = self._embed_doc(idx)
        torch.save(emb, path)
        return emb

    def compute(self, cache_embds=True) -> torch.Tensor:
        """
        Compute similarity matrix.
        """
        embeddings = []
        for i in tqdm(self.test_dataset["indices"][0:50] if DEBUG else self.test_dataset["indices"], desc=f"Computing {self.name} embeddings"):
            embeddings.append(self.compute_doc(i, force=not cache_embds))
        embeddings = torch.vstack(embeddings)
        sim_matrix = embeddings @ embeddings.T
        return sim_matrix#self.normalize(sim_matrix)

   


    @staticmethod
    def normalize(sim_matrix: torch.Tensor) -> torch.Tensor:
        return F.normalize(sim_matrix, dim=-1)

# ----------------- Gradient -----------------
class GradientRepresentation(BaseRepresentation):
    def __init__(self, estimator, name="GradientRepresentation"):
        super().__init__(name,
                         estimator.train_dataset, 
                         estimator.train_dataset_name, 
                         estimator.train_dataset_split, 
                         estimator.test_dataset, 
                         estimator.test_dataset_name, 
                         estimator.test_dataset_split, 
                         estimator.model_path,
                         )
        self.estimator = estimator
    def _embed_doc(self, idx: int) -> torch.Tensor:
        gradient = self.estimator.get_gradient(
            self.estimator.test_dataset_name,
            self.test_dataset_split,
            [idx]
        )[0].float()
        gradient = F.normalize(gradient.unsqueeze(0), p=2, dim=1)
        return gradient.squeeze(0).cpu()




# ----------------- Full Model -----------------
class FullModelRepresentation(BaseRepresentation):
    def __init__(self, 
                 model, 
                 tokenized_dataset, 
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 model_path,
                 name="full_model"):
        super().__init__(
                         name,
                         train_dataset, 
                         train_dataset_name, 
                         train_dataset_split, 
                         test_dataset, 
                         test_dataset_name, 
                         test_dataset_split, 
                         model_path
                         )
        self.model = model
        self.tokenized_dataset = tokenized_dataset

    def _embed_doc(self, idx: int) -> torch.Tensor:
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


from sentence_transformers import SentenceTransformer

# ----------------- SBERT -----------------
class SBERTRepresentation(BaseRepresentation):
    def __init__(self, 
                 model_path, 
                 decoded_texts, 
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 
                 ):
     
        self.st_model = SentenceTransformer(model_path)
        self.decoded_texts = decoded_texts
        super().__init__(
                         f"sbert_{model_path}",
                         train_dataset, 
                         train_dataset_name, 
                         train_dataset_split, 
                         test_dataset, 
                         test_dataset_name, 
                         test_dataset_split,    
                         model_path                
                         )
        

    def _embed_doc(self, idx: int) -> torch.Tensor:
        device = "cuda"
        self.st_model.to(device)
        text = self.decoded_texts[idx]
        emb = self.st_model.encode([text])[0]
        return F.normalize(torch.tensor(emb, dtype=torch.float32).unsqueeze(0), p=2, dim=1).squeeze(0).cpu()




# ----------------- BM25 -----------------
class BM25Representation(BaseRepresentation):
    def __init__(self, 
                 tokenized_docs,
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 name="bm25"):
        super().__init__(
            name,
            train_dataset,
            train_dataset_name,
            train_dataset_split,
            test_dataset,
            test_dataset_name,
            test_dataset_split,
            "-"
        )
        self.tokenized_docs = tokenized_docs
        self.bm25 = BM25Okapi(tokenized_docs)

    def _embed_doc(self, idx: int) -> torch.Tensor:
        scores = self.bm25.get_scores(self.tokenized_docs[idx])
        return F.normalize(torch.tensor(scores, dtype=torch.float32).unsqueeze(0), p=2, dim=1).squeeze(0).cpu()




# ----------------- TFIDF -----------------
class TFIDFRepresentation(BaseRepresentation):
    def __init__(self, 
                 tokenized_docs,
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 name="tfidf"):
        super().__init__(
            name,
            train_dataset,
            train_dataset_name,
            train_dataset_split,
            test_dataset,
            test_dataset_name,
            test_dataset_split,
            "-"
        )
        self.tokenized_docs = tokenized_docs
        self.docs_as_strings = [" ".join(map(str, doc)) for doc in tokenized_docs]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.docs_as_strings)

    def _embed_doc(self, idx: int) -> torch.Tensor:
        vec = torch.tensor(self.tfidf_matrix[idx].toarray(), dtype=torch.float32)
        return F.normalize(vec, p=2, dim=1).squeeze(0).cpu()




# ----------------- Jaccard -----------------
class JaccardRepresentation(BaseRepresentation):
    def __init__(self, 
                 tokenized_docs,
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 name="jaccard"):
        super().__init__(
            name,
            train_dataset,
            train_dataset_name,
            train_dataset_split,
            test_dataset,
            test_dataset_name,
            test_dataset_split,
            "-"
        )
        self.tokenized_docs = tokenized_docs

    def _embed_doc(self, idx: int) -> torch.Tensor:
        set_i = set(self.tokenized_docs[idx])
        n = len(self.tokenized_docs)
        vec = torch.zeros(n, dtype=torch.float32)
        for j in range(n):
            set_j = set(self.tokenized_docs[j])
            union = set_i | set_j
            vec[j] = 0.0 if len(union) == 0 else len(set_i & set_j) / len(union)
        return F.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0).cpu()




# ----------------- Normalized Attention Map -----------------
import torch
import torch.nn.functional as F
from tqdm import tqdm

class NormalizedAttentionMapRepresentation(BaseRepresentation):
    def __init__(self, 
                 model, 
                 tokenized_dataset, 
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 model_path,
                 max_seq_len=512, 
                 top_k_eig=128,
                 name="spectral_attention"):
        super().__init__(
            name,
            train_dataset,
            train_dataset_name,
            train_dataset_split,
            test_dataset,
            test_dataset_name,
            test_dataset_split,
            model_path
        )
        self.model = model
        self.tokenized_dataset = tokenized_dataset
        self.max_seq_len = max_seq_len
        self.top_k_eig = top_k_eig

    def _embed_doc(self, idx: int) -> torch.Tensor:
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


# ----------------- RepSim emulating https://github.com/plumprc/Failures-of-Influence-Functions-in-LLMs/blob/93f9489872c0ee33d05c7b7316f48f6256e47a86/repsim.py -----------------
class RepSimRepresentation(BaseRepresentation):
    def __init__(self, 
                 model, 
                 tokenized_dataset, 
                 train_dataset, 
                 train_dataset_name, 
                 train_dataset_split, 
                 test_dataset, 
                 test_dataset_name, 
                 test_dataset_split, 
                 model_path,
                 name="rep_sim"):
        super().__init__(
            name,
            train_dataset,
            train_dataset_name,
            train_dataset_split,
            test_dataset,
            test_dataset_name,
            test_dataset_split,
            model_path
        )
        self.model = model
        self.tokenized_dataset = tokenized_dataset

    def _embed_doc(self, idx: int) -> torch.Tensor:
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

import pickle
from collections import defaultdict
class DataInfRepresentation(BaseRepresentation):
    def __init__(self, estimator):
        super().__init__("DataInfRepresentation", 
                         estimator.train_dataset, 
                         estimator.train_dataset_name, 
                         estimator.train_dataset_split, 
                         estimator.test_dataset, estimator.test_dataset_name, 
                         estimator.test_dataset_split, 
                         estimator.model_path)
        self.estimator = estimator
        self._hvp_dict = None
        
        
    @property
    def hvp_dict(self):
        if self._hvp_dict is None:
            self.compute_hvp_approx_with_cache()
        return self._hvp_dict
    def compute_hvp_approx_with_cache(
        self,
        lambda_const_param=10,
        batch_size=32,
        val_batch_size=1000,
        inner_batch_size=1,
    ):
        """
        Computes a Hessian-vector product (HVP) approximation for influence estimation.
        If a cached HVP exists, loads it instead of recomputing.
        """
        estimator = self.estimator
        # Cache path
        hvp_cache_path = os.path.join(
            estimator.gradient_out_dir, "hvp", estimator.train_dataset_name, estimator.train_dataset_split,
            estimator.test_dataset_name, estimator.test_dataset_split, "hvp.pt"
        )
        print(hvp_cache_path)
        os.makedirs(os.path.dirname(hvp_cache_path), exist_ok=True)

        # Load from cache if exists
        if os.path.exists(hvp_cache_path):
            print(f"Loading cached HVP from {hvp_cache_path}")
            with open(hvp_cache_path, "rb") as f:
                hvp_dict = pickle.load(f)
            self._hvp_dict = hvp_dict
            return 

        # Otherwise, compute HVP
        val_grad_dict = estimator.load_gradients(estimator.test_dataset, estimator.test_dataset_name, estimator.test_dataset_split)
        hvp_dict = defaultdict(dict)
        n_train = len(val_grad_dict)
        val_ids = list(val_grad_dict.keys())
        weight_names = list(val_grad_dict[next(iter(val_grad_dict))].keys())
        
        # Compute per-parameter regularization constant
        print("Computing lambda_const...")
        sum_sq = torch.zeros(len(weight_names), dtype=torch.float64)
        for i in tqdm(range(0, n_train, batch_size), desc="Lambda_const computation"):
            batch_ids = list(range(i, min(i + batch_size, n_train)))
            batch_grads_list = estimator.get_gradient_fn(estimator.train_dataset_name, estimator.train_dataset_split, batch_ids)
            for grads_dict in batch_grads_list:
                per_weight_means = [grads_dict[w].view(-1).double().pow(2).mean() for w in weight_names]
                sum_sq += torch.tensor(per_weight_means, dtype=torch.float64)
        lambda_const = (sum_sq / n_train) / lambda_const_param
        lambda_const = lambda_const.to(dtype=torch.float32)

        # Precompute validation gradients stacked by weight
        val_grads_per_weight = [
            torch.stack([val_grad_dict[v][w].view(-1).float() for v in val_ids]) 
            for w in weight_names
        ]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for vb_start in tqdm(range(0, len(val_ids), val_batch_size), desc="Validation batches"):
            vb_end = min(vb_start + val_batch_size, len(val_ids))
            val_batch = val_ids[vb_start:vb_end]
            
            val_batch_grads_per_weight = [vg[vb_start:vb_end].to(device) for vg in val_grads_per_weight]
            hvp_batch_per_weight = [torch.zeros_like(vg) for vg in val_batch_grads_per_weight]
            lambda_const = lambda_const.to(device)

            # Iterate over training data in batches
            for i in range(0, n_train, batch_size):
                batch_ids = list(range(i, min(i + batch_size, n_train)))
                batch_grads_list = estimator.get_gradient_fn(estimator.train_dataset_name, estimator.train_dataset_split, batch_ids)

                batch_grads_per_weight = [
                    torch.stack([g[w].view(-1).float() for g in batch_grads_list]).to(device)
                    for w in weight_names
                ]

                batch_len = len(batch_ids)
                for k in range(0, batch_len, inner_batch_size):
                    sb_end = min(k + inner_batch_size, batch_len)

                    for w_idx, w in enumerate(weight_names):
                        val_w = val_batch_grads_per_weight[w_idx]        
                        tr_w = batch_grads_per_weight[w_idx][k:sb_end]   
                        
                        # Hessian-vector product approximation formula
                        sub_dot_w = val_w @ tr_w.t()                     
                        tr_norm_sq = tr_w.pow(2).sum(dim=1)             
                        denom_w = lambda_const[w_idx].view(()) + tr_norm_sq
                        sub_C_tmp_w = sub_dot_w / denom_w.unsqueeze(0)
                        contrib = (val_w.unsqueeze(1) - sub_C_tmp_w.unsqueeze(2) * tr_w.unsqueeze(0)).sum(dim=1)
                        denom_factor = (n_train * lambda_const[w_idx]).view(())
                        hvp_batch_per_weight[w_idx] += contrib / denom_factor

                del batch_grads_per_weight, batch_grads_list
                torch.cuda.empty_cache()

            # Store results
            for v_local_idx, val_id in enumerate(val_batch):
                for w_idx, w in enumerate(weight_names):
                    hvp_dict[val_id][w] = hvp_batch_per_weight[w_idx][v_local_idx].float().cpu()

        # Save to cache
        with open(hvp_cache_path, "wb") as f:
            pickle.dump(hvp_dict, f)

        self._hvp_dict = hvp_dict
    def _embed_doc(self, idx: int) -> torch.Tensor:
        # get inverse-Hessian times the validation gradient
        gradient =  torch.cat([self.hvp_dict[idx][k].flatten() for k in sorted(self.hvp_dict[idx].keys())])
        gradient = F.normalize(gradient.unsqueeze(0), p=2, dim=1)
        return gradient.squeeze(0).cpu().float()


