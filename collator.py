import torch
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DataCollatorWithPaddingMLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["masked_input_ids"], batch["masked_labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask,
            )
            if "golden_input_ids" in batch.keys():
                batch["masked_golden_input_ids"], batch["masked_golden_labels"] = self.torch_mask_tokens(
                    batch["golden_input_ids"], special_tokens_mask=special_tokens_mask,
                )
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True) for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels


@dataclass
class DataCollatorForCorrectionMLM:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    mlm: bool = True
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], _ = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask,
            )
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        labels = inputs.clone()
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True) for val in inputs.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels
    
    class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim (b 768)                     key and value
            video_embeds: num_vids x num_frames x embed_dim (b 8 768)          query
        Output
            o: num_vids x num_texts x embed_dim (b b 768)
        """


        # 我改的
        num_vids, num_frames, _ = video_embeds.shape 
        q = self.q_proj(video_embeds)
        q=q.reshape(num_vids, num_frames, self.num_heads, self.head_dim) # b 8 12 64

        # 我改的
        num_texts, _ = text_embeds.shape
        k = self.k_proj(text_embeds).unsqueeze(1) # b 1 768
        v = self.v_proj(text_embeds).unsqueeze(1) # b 1 768
        k = k.reshape(num_texts, 1, self.head_dim, self.num_heads) # b 1 64 12
        v = v.reshape(num_texts, 1, self.num_heads, self.head_dim) # b 1 12 64

        att_weight = F.softmax(torch.matmul(q, k)/math.sqrt(self.head_dim),dim=2) # b 8 12 12
        attention = att_weight@v # b 8 12 64
        attention = attention.reshape(num_texts, num_frames, self.embed_dim) # b 8 768
        o = self.out_proj(attention) # b 8 768

        return o


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        original_video_embeds = video_embeds # b 8 196
        text_embeds = self.layer_norm1(text_embeds) # key,value     b 196
        video_embeds = self.layer_norm1(video_embeds) # query     b 8 196

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)+original_video_embeds
        # attn_out = self.layer_norm2(attn_out)
        attn_out = attn_out+self.dropout(self.linear_proj(self.layer_norm2(attn_out)))
        # linear_out = self.linear_proj(attn_out)
        # out = attn_out + self.dropout(linear_out)
        # out = self.layer_norm3(out)

        return attn_out

