#!/usr/bin/python3
"""
Recipe for training speaker embeddings using the VoxCeleb dataset.

Modified for:
TDNN + BiLSTM + Multi-Head Attention Pooling + AAM-Softmax

Run:
    python train_speaker_embeddings.py hparams/train_tdnn_bilstm_mha_aam.yaml
"""

import os
import random
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training."""

    def compute_forward(self, batch, stage):
        """Forward pass: wav -> features -> embedding -> classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Optional waveform augmentation during training
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, lens = self.hparams.wav_augment(wavs, lens)

        # Feature extraction
        if (
            hasattr(self.hparams, "use_tacotron2_mel_spec")
            and self.hparams.use_tacotron2_mel_spec
        ):
            feats = self.hparams.compute_features(audio=wavs)
            feats = torch.transpose(feats, 1, 2)
        else:
            feats = self.modules.compute_features(wavs)

        # Feature normalization
        feats = self.modules.mean_var_norm(feats, lens)

        # Embedding model
        # Supports both signatures:
        #   embedding_model(feats)
        #   embedding_model(feats, lens)
        try:
            embeddings = self.modules.embedding_model(feats, lens)
        except TypeError:
            embeddings = self.modules.embedding_model(feats)

        # Classifier head
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute AAM-Softmax loss using speaker labels."""
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Replicate labels if augmentation concatenates examples
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            spkid = self.hparams.wav_augment.replicate_labels(spkid)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        # Per-batch LR update if scheduler supports it
        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Called at the beginning of each stage."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Called at the end of each stage."""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    """Creates datasets and data processing pipelines."""

    data_folder = hparams["data_folder"]

    # Load CSV datasets
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_annotation"],
        replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_annotation"],
        replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]

    # Label encoder
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        """Load waveform or random chunk."""
        duration = float(duration)

        if hparams["random_chunk"]:
            duration_sample = int(duration * hparams["sample_rate"])

            if duration_sample <= snt_len_sample:
                start_sample = 0
                stop_sample = duration_sample
            else:
                start_sample = random.randint(0, duration_sample - snt_len_sample)
                stop_sample = start_sample + snt_len_sample
        else:
            start_sample = int(start)
            stop_sample = int(stop)

        num_frames = max(1, stop_sample - start_sample)

        sig, fs = torchaudio.load(
            wav,
            num_frames=num_frames,
            frame_offset=start_sample,
        )

        # Convert [channels, time] -> [time]
        sig = sig.transpose(0, 1).squeeze(1)

        # Safety: resample only if needed
        if fs != hparams["sample_rate"]:
            sig = torchaudio.functional.resample(
                sig.unsqueeze(0), fs, hparams["sample_rate"]
            ).squeeze(0)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("spk_id")
    @sb.utils.data_pipeline.provides("spk_id", "spk_id_encoded")
    def label_pipeline(spk_id):
        """Encode speaker label."""
        yield spk_id
        yield label_encoder.encode_label_torch(spk_id)

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # Create/load label encoder from training set
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="spk_id",
    )

    # Output keys used by the Brain
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "spk_id_encoded"],
    )

    return train_data, valid_data, label_encoder


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Parse CLI args
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # DDP init
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hparams
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory early
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Download verification split locally if it is a URL
    verification_source = hparams["verification_file"]
    if verification_source.startswith("http://") or verification_source.startswith(
        "https://"
    ):
        veri_file_path = os.path.join(
            hparams["save_folder"], os.path.basename(verification_source)
        )
        download_file(verification_source, veri_file_path)
    else:
        veri_file_path = verification_source

    # Dataset CSV preparation
    from voxceleb_prepare import prepare_voxceleb

    run_on_main(
        prepare_voxceleb,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_folder": hparams["save_folder"],
            "verification_pairs_file": veri_file_path,
            "splits": ["train", "dev"],
            "split_ratio": hparams["split_ratio"],
            "seg_dur": hparams["sentence_len"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Optional augmentation dataset preparation
    if "prepare_noise_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_noise_data"])

    if "prepare_rir_data" in hparams:
        sb.utils.distributed.run_on_main(hparams["prepare_rir_data"])

    # Build datasets
    train_data, valid_data, label_encoder = dataio_prep(hparams)

    # Initialize brain
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Train
    speaker_brain.fit(
        epoch_counter=speaker_brain.hparams.epoch_counter,
        train_set=train_data,
        valid_set=valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )