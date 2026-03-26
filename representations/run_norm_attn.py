# python3 -m representations.run_norm_attn --model ./models/OLMo-2-0425-1B_ftrace_date-lm_conversational_lr1e-05_seed42 --train_dataset_name loris3/ftrace_date-lm_conversational --train_dataset_split train --test_dataset_split train --test_dataset_name loris3/ftrace_date-lm_conversational
import argparse
from utils import load_lora_model
import torch
import os
import pandas as pd
from representations.main import BaseRepresentation
from representations.common_args import add_common_args
from influence_estimation.estimator import BaseEstimator
import torch.nn.functional as F
from influence_estimation.util import tokenize_dataset

from representations.load_data import (   
    load_data_and_estimators)


class NormalizedAttentionMapEstimator(BaseEstimator):
    def __init__(self, model_path, train_dataset, train_dataset_name, train_dataset_split,
                 test_dataset, test_dataset_name, test_dataset_split,
                 device="cpu", param_list=[], param_list_gradients=[], eval_mode=False):
        super().__init__(
            model_path, train_dataset, train_dataset_name, train_dataset_split,
            test_dataset, test_dataset_name, test_dataset_split,
            device=device, param_list=param_list, param_list_gradients=param_list_gradients,
            eval_mode=eval_mode
        )
    def get_config_string(self):
        return f"{self.__class__.__name__}: {os.path.basename(model_path)}"
    def init_model_and_tokenizer(self):
        pass
    def run(self):
        results = self.embeddings @ self.embeddings.T
        self.influence_estimate = pd.DataFrame(
            results,
            index=list(range(self.embeddings.shape[0])),
            columns=list(range(self.embeddings.shape[0]))
        )
        os.makedirs(os.path.dirname(self.influence_estimate_path), exist_ok=True)
        self.save()


class NormalizedAttentionMapRepresentation(BaseRepresentation):
    def __init__(self, model, tokenized_dataset, train_dataset, train_dataset_name,
                 train_dataset_split, test_dataset, test_dataset_name, test_dataset_split,
                 model_path, top_k_eig=128, name="spectral_attention", results_only=False):
        super().__init__(name, train_dataset, train_dataset_name, train_dataset_split,
                         test_dataset, test_dataset_name, test_dataset_split,
                         model_path, results_only=results_only)
        self.estimator = NormalizedAttentionMapEstimator(
                model_path=model_path,
                train_dataset=train_dataset,
                train_dataset_name=train_dataset_name,
                train_dataset_split=train_dataset_split,
                test_dataset=test_dataset,
                test_dataset_name=test_dataset_name,
                test_dataset_split=test_dataset_split,
                device="cpu",
                param_list=[],
                param_list_gradients=[],
                eval_mode=False
            )
        if not self.results_only:
            self.model = model
            self.tokenized_dataset = tokenized_dataset
            self.max_seq_len = model.config.max_position_embeddings
            print("Max sequence length:", self.max_seq_len)
            self.top_k_eig = top_k_eig
            self.device ="cuda"
            self.model.to(self.device).eval()
        self.estimator.embeddings = self.compute()

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
        # print("doc_vector", doc_vector)
        return doc_vector.cpu()
def main(args=None):
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args(args)
    print(args)
    train_dataset, test_dataset = load_data_and_estimators(args.train_dataset_name, 
                                                           args.test_dataset_name, 
                                                           args.train_dataset_split, 
                                                           args.test_dataset_split)
    
    model, tokenizer = load_lora_model(args.model) if not args.results_only else (None, None)

    tokenized_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
        num_proc=10,
        re_index=False,
    ) if not args.results_only else None

    rep = NormalizedAttentionMapRepresentation(
                 model,
                 tokenized_dataset, 
                 train_dataset, 
                 args.train_dataset_name, 
                 args.train_dataset_split, 
                 test_dataset, 
                 args.test_dataset_name, 
                 args.test_dataset_split, 
                 args.model,
                 results_only=args.results_only, 
        )

    del train_dataset, test_dataset, tokenizer, model
    return rep.estimator.embeddings, rep.get_influence_estimate()

   
if __name__ == "__main__":
    main()
    
    
