import torch
import torch.nn as nn


class TDNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AudioTDNNBiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim=80,
        tdnn_channels=(256, 256, 256),
        tdnn_kernel_sizes=(5, 3, 3),
        lstm_hidden=512,
        lstm_layers=1,
        proj_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.tdnn1 = TDNNBlock(input_dim, tdnn_channels[0], tdnn_kernel_sizes[0])
        self.tdnn2 = TDNNBlock(tdnn_channels[0], tdnn_channels[1], tdnn_kernel_sizes[1])
        self.tdnn3 = TDNNBlock(tdnn_channels[1], tdnn_channels[2], tdnn_kernel_sizes[2])

        self.bilstm = nn.LSTM(
            input_size=tdnn_channels[2],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.proj = nn.Linear(lstm_hidden * 2, proj_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, lengths=None):
        """
        feats: [B, T, 80]
        returns: [B, T, D]
        """
        x = feats.transpose(1, 2)
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = x.transpose(1, 2)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            x, _ = self.bilstm(x)

        x = self.dropout(x)
        x = self.proj(x)
        return x
