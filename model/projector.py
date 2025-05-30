import torch
import torch.nn as nn

class Projector(nn.Module):
    def __init__(self,
                 input_patch_dim,        # CLAP 중간 레이어의 특징 차원 (예: 512)
                 num_input_patches,      # CLAP 중간 레이어의 패치 수 (예: 256)
                 output_seq_len,         # 목표 시퀀스 길이 (n_query, 예: 256 또는 64)
                 output_embed_dim,       # 목표 임베딩 차원 (Emu 비주얼 토큰 차원, 예: 1792)
                 projector_transformer_hidden_dim, # 프로젝터 내부 Transformer의 d_model (예: 768)
                 projector_num_transformer_layers, # 총 Transformer 레이어 수
                 projector_num_heads,    # 어텐션 헤드 수
                 projector_dropout):     # 드롭아웃 비율
        super(Projector, self).__init__()

        self.input_patch_dim = input_patch_dim
        self.num_input_patches = num_input_patches
        self.output_seq_len = output_seq_len
        self.output_embed_dim = output_embed_dim
        self.hidden_dim = projector_transformer_hidden_dim
        self.num_layers = projector_num_transformer_layers

        # 1. 입력 패치 프로젝션: input_patch_dim -> hidden_dim
        self.input_proj = nn.Linear(input_patch_dim, self.hidden_dim)
        
        # 2. 입력 오디오 패치 시퀀스를 위한 위치 임베딩
        self.input_pos_embed = nn.Parameter(torch.randn(1, num_input_patches, self.hidden_dim))
        self.dropout = nn.Dropout(projector_dropout)

        if self.num_layers == 0:
            # Transformer 없이 MLP와 풀링으로 처리
            self.mlp_direct_map = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.output_embed_dim)
            )
            # AdaptiveAvgPool1d는 (B, C, L_in)을 기대하므로, (B, L_in, C)를 transpose 필요
            # 또는 MLP 후 (B, L_in, output_embed_dim)을 (B, output_embed_dim, L_in)으로 transpose 후 풀링
            # 여기서는 num_input_patches를 output_seq_len으로 직접 매핑 시도
            if num_input_patches != output_seq_len:
                 self.final_seq_len_adjust_mlp = nn.Linear(num_input_patches, output_seq_len)

        elif self.num_layers == 1:
            # 인코더만 1개 사용
            encoder_layer_config = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=projector_num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=projector_dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            self.audio_encoder = nn.TransformerEncoder(encoder_layer_config, num_layers=1)
            # 인코더 출력 (B, num_input_patches, hidden_dim)을 (B, output_seq_len, output_embed_dim)으로 매핑
            self.encoder_output_proj = nn.Linear(self.hidden_dim, self.output_embed_dim)
            if num_input_patches != output_seq_len:
                # (B, num_input_patches, output_embed_dim) -> (B, output_embed_dim, num_input_patches)
                # -> AdaptiveAvgPool1d -> (B, output_embed_dim, output_seq_len) -> (B, output_seq_len, output_embed_dim)
                self.final_seq_len_adjust_pool = nn.AdaptiveAvgPool1d(output_seq_len)
        
        else: # num_layers >= 2: 인코더-디코더 구조
            num_encoder_layers = self.num_layers // 2
            num_decoder_layers = self.num_layers - num_encoder_layers

            encoder_layer_config = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=projector_num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=projector_dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            self.audio_encoder = nn.TransformerEncoder(encoder_layer_config, num_layers=num_encoder_layers)

            # 학습 가능한 출력 쿼리 토큰 (목표 시퀀스 길이, 목표 임베딩 차원)
            self.output_query_tokens = nn.Parameter(torch.randn(1, output_seq_len, self.output_embed_dim))
            
            # 디코더의 d_model은 output_embed_dim을 사용
            # 메모리(인코더 출력)의 차원이 hidden_dim이므로, 이를 output_embed_dim으로 프로젝션 필요
            self.encoder_output_to_decoder_memory_proj = nn.Identity()
            if self.hidden_dim != self.output_embed_dim:
                 self.encoder_output_to_decoder_memory_proj = nn.Linear(self.hidden_dim, self.output_embed_dim)

            decoder_layer_config = nn.TransformerDecoderLayer(
                d_model=self.output_embed_dim, 
                nhead=projector_num_heads,
                dim_feedforward=self.output_embed_dim * 4,
                dropout=projector_dropout,
                activation="gelu", batch_first=True, norm_first=True
            )
            self.output_decoder = nn.TransformerDecoder(decoder_layer_config, num_layers=num_decoder_layers)

    def forward(self, audio_patch_sequence): 
        # audio_patch_sequence: [B, num_input_patches, input_patch_dim]
        B = audio_patch_sequence.size(0)

        # 1. 입력 프로젝션 및 위치 임베딩
        # [B, num_input_patches, input_patch_dim] -> [B, num_input_patches, hidden_dim]
        projected_patches = self.input_proj(audio_patch_sequence) 
        patches_with_pos = projected_patches + self.input_pos_embed
        x = self.dropout(patches_with_pos)

        if self.num_layers == 0:
            # Transformer 없이 MLP와 풀링으로 처리
            # x: [B, num_input_patches, hidden_dim]
            x = self.mlp_direct_map(x) # -> [B, num_input_patches, output_embed_dim]
            if self.num_input_patches != self.output_seq_len:
                # (B, L_in, C) -> (B, C, L_in)
                x = x.transpose(1, 2) # -> [B, output_embed_dim, num_input_patches]
                x = self.final_seq_len_adjust_mlp(x) # Linear는 마지막 차원에 작용 -> [B, output_embed_dim, output_seq_len]
                x = x.transpose(1, 2) # -> [B, output_seq_len, output_embed_dim]
            output_embeddings = x

        elif self.num_layers == 1:
            # 인코더만 사용
            # x: [B, num_input_patches, hidden_dim]
            encoded_audio = self.audio_encoder(x) # -> [B, num_input_patches, hidden_dim]
            # -> [B, num_input_patches, output_embed_dim]
            projected_encoded_audio = self.encoder_output_proj(encoded_audio) 
            
            if self.num_input_patches != self.output_seq_len:
                # (B, L_in, C_out) -> (B, C_out, L_in)
                x_for_pool = projected_encoded_audio.transpose(1, 2) 
                # -> (B, C_out, L_out)
                pooled_audio = self.final_seq_len_adjust_pool(x_for_pool) 
                # -> (B, L_out, C_out)
                output_embeddings = pooled_audio.transpose(1, 2) 
            else:
                output_embeddings = projected_encoded_audio
        
        else: # num_layers >= 2: 인코더-디코더 구조
            # x: [B, num_input_patches, hidden_dim]
            encoded_audio_patches = self.audio_encoder(x) # -> [B, num_input_patches, hidden_dim]

            # 디코더 메모리 준비: [B, num_input_patches, hidden_dim] -> [B, num_input_patches, output_embed_dim]
            decoder_memory = self.encoder_output_to_decoder_memory_proj(encoded_audio_patches)
            
            # 출력 쿼리 토큰 준비
            tgt_queries = self.output_query_tokens.repeat(B, 1, 1) # -> [B, output_seq_len, output_embed_dim]
            
            # 디코더 통과
            output_embeddings = self.output_decoder(tgt=tgt_queries, memory=decoder_memory) # -> [B, output_seq_len, output_embed_dim]

        return output_embeddings