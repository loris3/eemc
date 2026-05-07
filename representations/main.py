import torch
import logging

logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
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
                 model_path, cache_dir: str = "./cache/representations", results_only=False, normalize_vectors=False, func_only=False):
        self.name = name
        self.func_only = func_only
        self.train_dataset = train_dataset
        self.train_dataset_name = train_dataset_name
        self.train_dataset_split = train_dataset_split
        self.test_dataset = test_dataset
        self.test_dataset_name = test_dataset_name
        self.test_dataset_split = test_dataset_split
        self.model_path = model_path
        self.cache_dir = os.path.join(
            cache_dir,
            self.name,
            train_dataset_name,
            train_dataset_split,
            test_dataset_name,
            test_dataset_split,
            model_path,
        )

        self.local_cache_dir = os.path.join(
            "/tmp",
            cache_dir.strip("./"),
            self.name,
            train_dataset_name,
            train_dataset_split,
            test_dataset_name,
            test_dataset_split,
            model_path,
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.local_cache_dir, exist_ok=True)
        self.results_only = results_only
        self.normalize_vectors = normalize_vectors
        if self.results_only and not self.func_only:
            self.copy_cache_to_tmp()
           

    def get_influence_estimate(self):
        self.estimator.run_cached()
        return self.estimator.influence_estimate
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

           
                        with tqdm(total=len(futures), desc="Copying files (rep)") as pbar:
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
        local_path = os.path.join(self.local_cache_dir, f"doc_{idx}.pt")
        remote_path = os.path.join(self.cache_dir, f"doc_{idx}.pt")
        if not force and os.path.exists(local_path):
            try:
                return torch.load(local_path, map_location="cpu")
            except Exception:
                pass

        if not force and os.path.exists(remote_path):
            try:
                return torch.load(remote_path, map_location="cpu")
            except Exception:
                pass

        if self.results_only:
            raise RuntimeError(
                f"results_only=True but cache not found for doc {idx}"
            )

        emb = self._embed_doc(idx)
        os.makedirs(os.path.dirname(remote_path), exist_ok=True)
        torch.save(emb, remote_path)

        return emb
    def compute(self, cache_embds=True) -> torch.Tensor:
        indices = self.test_dataset["indices"][0:DEBUG] if DEBUG else self.test_dataset["indices"]

        embeddings = [None] * len(indices) #if self.results_only else None

        with tqdm(total=len(indices), desc=f"Computing {self.name} embeddings") as pbar:
            for pos, i in enumerate(indices):
                emb = self.compute_doc(i, force=not cache_embds)
                embeddings[pos] = emb
                pbar.update(1)

        # if self.results_only:
        embeddings = torch.vstack(embeddings)
        if self.normalize_vectors:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

