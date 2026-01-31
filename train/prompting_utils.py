from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


import torch
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 max_prompt_len=8000, max_gen_length=377, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_gen_length = max_gen_length
        self.max_prompt_len = max_prompt_len


    # language modeling
    def lm_prompt(self, text_ids_pairs):
        prompts_list, responses_list = text_ids_pairs
        pad_id = self.text_tokenizer.pad_token_id

        # 计算每条序列的总长度 = prompt + response + eos
        if responses_list.shape[1] < self.max_gen_length:
            max_seq_len = prompts_list.shape[1] + responses_list.shape[1]
        else:
            max_seq_len = prompts_list.shape[1] + self.max_gen_length

        sequence_ids = []
        attention_masks = []
        label_ids = []

        for prompt_ids, resp_ids in zip(prompts_list, responses_list):
            prompt_ids = prompt_ids.tolist()
            resp_ids   = resp_ids.tolist()

            # 拼接 prompt + response + EOS
            temp_ids = prompt_ids + resp_ids
            temp_masks = [1] * len(temp_ids)
            temp_labels = temp_ids.copy()

            # padding 或截断到 max_seq_len
            if len(temp_ids) < max_seq_len:
                pad_len = max_seq_len - len(temp_ids)
                temp_ids.extend([pad_id] * pad_len)
                temp_labels.extend([pad_id] * pad_len)
                temp_masks.extend([0] * pad_len)
            else:
                temp_ids = temp_ids[:max_seq_len]
                temp_labels = temp_labels[:max_seq_len]
                temp_masks = temp_masks[:max_seq_len]

            # 转为张量并累积
            sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
            attention_masks.append(torch.tensor(temp_masks).unsqueeze(0))
            label_ids.append(torch.tensor(temp_labels).unsqueeze(0))

        input_ids = torch.cat(sequence_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label_ids = torch.cat(label_ids, dim=0)
        

        return input_ids, label_ids, prompts_list.shape[1]

        
    

    def mask_prompt(self):
        pass

    def __call__(self, input):
        prompts, responses = input

        enc = self.text_tokenizer(
            prompts,
            padding=False,
            truncation=False,
            return_length=True
        )
        lengths = enc["length"]
        # 2) 过滤出长度 <= max_len 的 indices
        keep_indices = [i for i, L in enumerate(lengths) if L <= self.max_prompt_len]
        drop_num = len(prompts) - len(keep_indices)
        
        prompts  = [prompts[i]  for i in keep_indices]
        responses = [responses[i] for i in keep_indices]

        # 使用 tokenizer 将 raw text 转为 token ids
        prompt_ids = self.text_tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            padding_side = "left"
        )['input_ids']
        response_ids = self.text_tokenizer(
            responses,
            padding=True,
            return_tensors="pt",
            padding_side = "right"
        )['input_ids']
        input_ids_lm, labels_lm, start_pos = self.lm_prompt((prompt_ids, response_ids))
        return input_ids_lm, labels_lm, start_pos, drop_num

import torch

class SegmentedPrompting:
    def __init__(self, tokenizer, max_prompt_len=8000, max_gen_length=377):
        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_gen_length = max_gen_length
        self.pad_id = tokenizer.pad_token_id

    def get_batch_data(self, input_list, target_list, target_weights_list, 
                       src_template, suffix_template):
        
        batch_input_ids = []
        batch_labels = []
        batch_weights = []
        batch_start_pos = []
        keep_indices = []
        batch_response_lens = []
        
        for i, (inp, tgt) in enumerate(zip(input_list, target_list)):
            # Tokenize Prompt
            prompt_text = src_template.replace("{{src_code}}", inp)
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
            
            if len(prompt_ids) > self.max_prompt_len:
                continue
                
            # Tokenize Target
            target_ids = self.tokenizer(tgt, add_special_tokens=False).input_ids
            
            # Tokenize Suffix
            suffix_ids = self.tokenizer(suffix_template, add_special_tokens=False).input_ids
            
            # Weights
            w = target_weights_list[i] if target_weights_list else []
            t_weights = torch.ones(len(target_ids), dtype=torch.float32)
            if w:
                assert len(w) == len(target_ids), f"len(w): {len(w)}, len(target_ids): {len(target_ids)}"
                t_weights[:len(target_ids)] += torch.tensor(w[:len(target_ids)], dtype=torch.float32)
            
            response_ids = target_ids + suffix_ids
            response_weights = torch.cat([t_weights, torch.ones(len(suffix_ids), dtype=torch.float32)])
            
            if len(response_ids) > self.max_gen_length:
                response_ids = response_ids[:self.max_gen_length]
                response_weights = response_weights[:self.max_gen_length]

            curr_input_ids = prompt_ids + response_ids
            curr_labels = [-100] * len(prompt_ids) + response_ids
            curr_weights = [1.0] * len(prompt_ids) + response_weights.tolist()
            
            batch_input_ids.append(torch.tensor(curr_input_ids))
            batch_labels.append(torch.tensor(curr_labels))
            batch_weights.append(torch.tensor(curr_weights))
            batch_start_pos.append(len(prompt_ids))
            keep_indices.append(i)
            batch_response_lens.append(len(response_ids))

        if not batch_input_ids:
            return None

        # Padding
        batch_max_prompt = max(batch_start_pos)
        
        padded_inputs = []
        padded_labels = []
        padded_weights = []
        
        for b in range(len(batch_input_ids)):
            p_len = batch_start_pos[b]
            pad_p_len = batch_max_prompt - p_len
            
            p_padding = torch.full((pad_p_len,), self.pad_id, dtype=torch.long)
            p_label_padding = torch.full((pad_p_len,), -100, dtype=torch.long)
            p_weight_padding = torch.ones((pad_p_len,), dtype=torch.float32)
            
            curr_ids = torch.cat([p_padding, batch_input_ids[b]])
            curr_lbl = torch.cat([p_label_padding, batch_labels[b]])
            curr_wgt = torch.cat([p_weight_padding, batch_weights[b]])
            
            padded_inputs.append(curr_ids)
            padded_labels.append(curr_lbl)
            padded_weights.append(curr_wgt)
            
        max_total = max([x.size(0) for x in padded_inputs])
        
        final_inputs = []
        final_labels = []
        final_weights = []
        
        for i in range(len(padded_inputs)):
            rem = max_total - padded_inputs[i].size(0)
            if rem > 0:
                r_pad = torch.full((rem,), self.pad_id, dtype=torch.long)
                r_lbl = torch.full((rem,), -100, dtype=torch.long)
                r_wgt = torch.ones((rem,), dtype=torch.float32)
                
                final_inputs.append(torch.cat([padded_inputs[i], r_pad]))
                final_labels.append(torch.cat([padded_labels[i], r_lbl]))
                final_weights.append(torch.cat([padded_weights[i], r_wgt]))
            else:
                final_inputs.append(padded_inputs[i])
                final_labels.append(padded_labels[i])
                final_weights.append(padded_weights[i])
                
        return (torch.stack(final_inputs), 
                torch.stack(final_labels), 
                torch.stack(final_weights), 
                batch_max_prompt, 
                keep_indices,
                batch_response_lens)


if __name__ == '__main__':
    pass