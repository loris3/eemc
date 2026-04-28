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
    def __init__(self, model,    tokenizer,tokenized_dataset, train_dataset, train_dataset_name,
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
            if "Qwen" in model_path or "Llama" in model_path:
                self.is_gqa = True
            elif "OLMo-2" in model_path or "Olmo-3-1025-7B" in model_path:
                self.is_gqa = False
            else:
                raise NotImplementedError
            self.tokenized_dataset = tokenized_dataset
            self.max_seq_len = tokenizer.model_max_length
            print("Max sequence length:", self.max_seq_len)
            self.top_k_eig = top_k_eig
            self.device ="cuda"
            self.model.to(self.device).eval()
        self.estimator.embeddings = self.compute()

    def _embed_doc(self, idx: int) -> torch.Tensor:
        import torch
        import torch.nn.functional as F

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

        base_model = getattr(self.model, "base_model", self.model)
        layers = base_model.model.model.layers
        
        L = len(outputs.attentions)
        norm_maps = torch.zeros(
            (L, self.max_seq_len, self.max_seq_len),device=self.device
        )
        num_bytes = norm_maps.element_size() * norm_maps.nelement()
        num_gb = num_bytes / (1024 ** 3)

        print(f"Tensor size: {num_gb:.6f} GB")
        chunk_size = 128
        for l, alpha_layer in enumerate(outputs.attentions):
            x_i = outputs.hidden_states[l][0, :seq_len_use, :]  
            attn = layers[l].self_attn
            
            alpha = alpha_layer[0, :, :seq_len_use, :seq_len_use] # multi-head attention weights

            num_heads, num_tokens, _ = alpha.shape
            d_model = x_i.size(-1)
            d_head = attn.head_dim
            
            # per-head value transformation 
            if not self.is_gqa:
                # for MHA (reshape into heads (h, j, d_h))
                v = x_i @ attn.v_proj.weight.T # no bias
                v = v.view(num_tokens, num_heads, d_head).permute(1, 0, 2)           
            else:
                # for GQA (expand KV to Q heads)
                # group_size = n_heads / n_kv_groups
                H_kv = attn.v_proj.out_features // d_head
                v = x_i @ attn.v_proj.weight.T # no bias
                v = v.view(seq_len_use, H_kv, d_head)
                v = v.repeat_interleave(num_heads // H_kv, dim=1)  
                v = v.permute(1, 0, 2)                         


            # print("d_model", d_model)
            # head contribution weighted by attention + multi-head integration as in Appendix A of Kobayashi et al. 2020
            W = attn.o_proj.weight.view(num_heads, d_head, d_model)

            
            
            h, i, j = alpha.shape
            m = W.shape[-1]

            # norm-based attribution as in Appendix A of Kobayashi et al. 2020
            ############### memory safe implementation: torch.norm(torch.einsum("hijd,hdm->ijm", A, W), dim=-1)
            block_size=256 # adjust depending on gpu and model size
            
            out = torch.zeros((i, j), device=alpha.device, dtype=alpha.dtype)

            # precompute per head (we want to avoid i,j,m materialization later)
            vw = torch.einsum("hjd,hdm->hjm", v, W)  # [h,j,m]

            for m0 in range(0, m, block_size):
                m1 = min(m0 + block_size, m)

                # [h, j, block]
                t_block = vw[:, :, m0:m1]

                # compute s_{ijb} = sum_h alpha[h,i,j] * t[h,j,b]
                # result: [i,j,block]
                s = torch.einsum("hij,hjb->ijb", alpha, t_block)

                # accumulate sum over block dimension
                out += (s ** 2).sum(dim=-1) # for norm below

             
            # print("out", out.shape,flush=True)
            
            
            norm_maps[l,0:seq_len_use,0:seq_len_use] = torch.sqrt(out) # norm 
            ###############################################################################    
            
             
        # print("norm_maps", norm_maps[0].shape)
        torch.manual_seed(42)
        _, s, _ = torch.svd_lowrank(norm_maps.to(self.device), q=self.top_k_eig)

        doc_vector = s[:,:min(self.top_k_eig, s.size(1))].flatten()
        print("doc_vec", doc_vector)
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
                 tokenizer,
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
    
    
