import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_encoder import AudioTDNNBiLSTMEncoder
from visual_encoder import VisualLipResNet18Encoder
from fusion import GatedFusion
from conformer_block import SharedConformerBlock
from pooling import MultiHeadAttentionPooling
from utils import lengths_to_mask


class AVSpeakerVerificationModel(nn.Module):
    def __init__(
        self,
        audio_input_dim=80,
        branch_dim=256,
        conformer_heads=4,
        pooling_heads=5,
        embedding_dim=192,
        visual_pretrained=False,
        visual_grayscale=True,
        dropout=0.1,
    ):
        super().__init__()

        self.audio_encoder = AudioTDNNBiLSTMEncoder(
            input_dim=audio_input_dim,
            proj_dim=branch_dim,
            dropout=dropout,
        )

        self.visual_encoder = VisualLipResNet18Encoder(
            output_dim=branch_dim,
            pretrained=visual_pretrained,
            grayscale=visual_grayscale,
        )

        self.gated_fusion = GatedFusion(branch_dim)

        self.shared_temporal_encoder = SharedConformerBlock(
            dim=branch_dim,
            num_heads=conformer_heads,
            dropout=dropout,
        )

        self.pooling = MultiHeadAttentionPooling(
            input_dim=branch_dim,
            num_heads=pooling_heads,
        )

        self.fc = nn.Linear(branch_dim * pooling_heads, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, audio_feats, video_frames, lengths=None):
        """
        audio_feats:  [B, T, 80]
        video_frames: [B, T, C, H, W]
        lengths:      [B]
        """
        audio_x = self.audio_encoder(audio_feats, lengths)
        visual_x = self.visual_encoder(video_frames)

        min_t = min(audio_x.size(1), visual_x.size(1))
        audio_x = audio_x[:, :min_t]
        visual_x = visual_x[:, :min_t]

        if lengths is not None:
            lengths = torch.clamp(lengths, max=min_t)

        fused = self.gated_fusion(audio_x, visual_x)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = ~lengths_to_mask(lengths, min_t)

        x = self.shared_temporal_encoder(fused, key_padding_mask=key_padding_mask)
        x = self.pooling(x, lengths)

        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def score_pairs(
        self,
        audio_feats_1,
        video_frames_1,
        audio_feats_2,
        video_frames_2,
        lengths_1=None,
        lengths_2=None,
    ):
        emb1 = self.forward(audio_feats_1, video_frames_1, lengths_1)
        emb2 = self.forward(audio_feats_2, video_frames_2, lengths_2)
        return F.cosine_similarity(emb1, emb2)


if __name__ == "__main__":
    batch_size = 2
    time_steps = 75
    feat_dim = 80
    h, w = 112, 112

    audio = torch.randn(batch_size, time_steps, feat_dim)
    video = torch.randn(batch_size, time_steps, 1, h, w)
    lengths = torch.tensor([75, 68])

    model = AVSpeakerVerificationModel()
    emb = model(audio, video, lengths)
    print("Embedding shape:", emb.shape)

    scores = model.score_pairs(audio, video, audio, video, lengths, lengths)
    print("Scores shape:", scores.shape)
