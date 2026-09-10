from transformers import ModernBertConfig

LADDER = {
    "17m": dict(num_hidden_layers=7, hidden_size=256, intermediate_size=384, num_attention_heads=4),
    "32m": dict(num_hidden_layers=10, hidden_size=384, intermediate_size=576, num_attention_heads=6),
    "68m": dict(num_hidden_layers=19, hidden_size=512, intermediate_size=768, num_attention_heads=8),
    "150m": dict(num_hidden_layers=22, hidden_size=768, intermediate_size=1152, num_attention_heads=12),
}

BASE_WIDTH = 256

VOCAB_SIZE = 50368
CLS_ID, SEP_ID, PAD_ID, MASK_ID = 50281, 50282, 50283, 50284


def make_config(size, seq_len):
    return ModernBertConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=seq_len,
        global_attn_every_n_layers=3,
        local_attention=128,
        global_rope_theta=160000.0,
        local_rope_theta=160000.0,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        classifier_bias=False,
        decoder_bias=True,
        tie_word_embeddings=True,
        sparse_prediction=True,
        pad_token_id=PAD_ID,
        bos_token_id=CLS_ID,
        eos_token_id=SEP_ID,
        cls_token_id=CLS_ID,
        sep_token_id=SEP_ID,
        **LADDER[size],
    )
