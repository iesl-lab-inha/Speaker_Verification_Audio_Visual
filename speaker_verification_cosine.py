#!/usr/bin/python3
"""
Speaker verification with cosine similarity on top of pretrained embeddings.

Modified for:
TDNN + BiLSTM + Multi-Head Attention Pooling + 192-dim L2-normalized embeddings

Run:
    python speaker_verification_cosine.py hparams/verification_tdnn_bilstm_mha.yaml
"""

import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm import tqdm

import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.utils.metric_stats import EER, minDCF


def compute_embedding(wavs, wav_lens):
    """Compute speaker embeddings from waveforms.

    Arguments
    ---------
    wavs : torch.Tensor
        Waveforms of shape [batch, time]
    wav_lens : torch.Tensor
        Relative lengths of shape [batch]

    Returns
    -------
    torch.Tensor
        Embeddings of shape [batch, emb_dim]
    """
    with torch.no_grad():
        feats = params["compute_features"](wavs)
        feats = params["mean_var_norm"](feats, wav_lens)

        # Support both signatures:
        #   embedding_model(feats)
        #   embedding_model(feats, wav_lens)
        try:
            embeddings = params["embedding_model"](feats, wav_lens)
        except TypeError:
            embeddings = params["embedding_model"](feats)

        # Ensure [B, D]
        if embeddings.dim() == 3 and embeddings.size(1) == 1:
            embeddings = embeddings.squeeze(1)

    return embeddings


def compute_embedding_loop(data_loader):
    """Compute embeddings for all utterances in a dataloader."""
    embedding_dict = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, dynamic_ncols=True):
            batch = batch.to(run_opts["device"])
            seg_ids = batch.id
            wavs, lens = batch.sig

            # Skip already computed ids
            if all(seg_id in embedding_dict for seg_id in seg_ids):
                continue

            wavs = wavs.to(run_opts["device"])
            lens = lens.to(run_opts["device"])

            emb = compute_embedding(wavs, lens)

            for i, seg_id in enumerate(seg_ids):
                embedding_dict[seg_id] = emb[i].detach().clone()

    return embedding_dict


def get_verification_scores(veri_test):
    """Compute positive and negative verification scores."""
    positive_scores = []
    negative_scores = []

    save_file = os.path.join(params["output_folder"], "scores.txt")
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    if "score_norm" in params:
        train_cohort = torch.stack(list(train_dict.values()))  # [N, D]

    with open(save_file, "w", encoding="utf-8") as s_file:
        for line in veri_test:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            lab_pair = int(parts[0].rstrip().split(".")[0].strip())
            enrol_id = parts[1].rstrip().split(".")[0].strip()
            test_id = parts[2].rstrip().split(".")[0].strip()

            if enrol_id not in enrol_dict or test_id not in test_dict:
                continue

            enrol = enrol_dict[enrol_id]  # [D]
            test = test_dict[test_id]     # [D]

            # Base cosine score
            score = similarity(enrol.unsqueeze(0), test.unsqueeze(0))[0]

            if "score_norm" in params:
                # enrol-vs-cohort
                enrol_rep = enrol.unsqueeze(0).repeat(train_cohort.shape[0], 1)
                score_e_c = similarity(enrol_rep, train_cohort)

                if "cohort_size" in params:
                    k = min(params["cohort_size"], score_e_c.shape[0])
                    score_e_c = torch.topk(score_e_c, k=k, dim=0)[0]

                mean_e_c = torch.mean(score_e_c, dim=0)
                std_e_c = torch.std(score_e_c, dim=0).clamp_min(1e-6)

                # test-vs-cohort
                test_rep = test.unsqueeze(0).repeat(train_cohort.shape[0], 1)
                score_t_c = similarity(test_rep, train_cohort)

                if "cohort_size" in params:
                    k = min(params["cohort_size"], score_t_c.shape[0])
                    score_t_c = torch.topk(score_t_c, k=k, dim=0)[0]

                mean_t_c = torch.mean(score_t_c, dim=0)
                std_t_c = torch.std(score_t_c, dim=0).clamp_min(1e-6)

                # Score normalization
                if params["score_norm"] == "z-norm":
                    score = (score - mean_e_c) / std_e_c
                elif params["score_norm"] == "t-norm":
                    score = (score - mean_t_c) / std_t_c
                elif params["score_norm"] == "s-norm":
                    score_e = (score - mean_e_c) / std_e_c
                    score_t = (score - mean_t_c) / std_t_c
                    score = 0.5 * (score_e + score_t)

            s_file.write(f"{enrol_id} {test_id} {lab_pair} {float(score):.6f}\n")

            if lab_pair == 1:
                positive_scores.append(float(score))
            else:
                negative_scores.append(float(score))

    return positive_scores, negative_scores


def dataio_prep(params):
    """Create dataloaders and audio pipelines."""
    data_folder = params["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["train_data"],
        replacements={"data_root": data_folder},
    )
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        select_n=params["n_train_snts"],
    )

    enrol_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["enrol_data"],
        replacements={"data_root": data_folder},
    )
    enrol_data = enrol_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=params["test_data"],
        replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, enrol_data, test_data]

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop):
        start = int(start)
        stop = int(stop)
        num_frames = max(1, stop - start)

        sig, fs = torchaudio.load(
            wav,
            num_frames=num_frames,
            frame_offset=start,
        )

        sig = sig.transpose(0, 1).squeeze(1)

        if fs != params["sample_rate"]:
            sig = torchaudio.functional.resample(
                sig.unsqueeze(0), fs, params["sample_rate"]
            ).squeeze(0)

        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    train_dataloader = sb.dataio.dataloader.make_dataloader(
        train_data, **params["train_dataloader_opts"]
    )
    enrol_dataloader = sb.dataio.dataloader.make_dataloader(
        enrol_data, **params["enrol_dataloader_opts"]
    )
    test_dataloader = sb.dataio.dataloader.make_dataloader(
        test_data, **params["test_dataloader_opts"]
    )

    return train_dataloader, enrol_dataloader, test_dataloader


if __name__ == "__main__":
    logger = get_logger(__name__)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    # Parse args
    params_file, run_opts, overrides = sb.core.parse_arguments(sys.argv[1:])

    with open(params_file, encoding="utf-8") as fin:
        params = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=params["output_folder"],
        hyperparams_to_save=params_file,
        overrides=overrides,
    )

    # Resolve verification file
    verification_source = params["verification_file"]
    if verification_source.startswith("http://") or verification_source.startswith(
        "https://"
    ):
        veri_file_path = os.path.join(
            params["save_folder"], os.path.basename(verification_source)
        )
        download_file(verification_source, veri_file_path)
    else:
        veri_file_path = verification_source

    # Prepare CSVs
    from voxceleb_prepare import prepare_voxceleb

    prepare_voxceleb(
        data_folder=params["data_folder"],
        save_folder=params["save_folder"],
        verification_pairs_file=veri_file_path,
        splits=["train", "dev", "test"],
        split_ratio=params["split_ratio"],
        seg_dur=3.0,
        skip_prep=params["skip_prep"],
        source=(params["voxceleb_source"] if "voxceleb_source" in params else None),
    )

    # Data
    train_dataloader, enrol_dataloader, test_dataloader = dataio_prep(params)

    # Load checkpoint / pretrained files
    run_on_main(params["pretrainer"].collect_files)
    params["pretrainer"].load_collected()

    params["embedding_model"].eval()
    params["embedding_model"].to(run_opts["device"])

    logger.info("Computing enrollment and test embeddings...")

    enrol_dict = compute_embedding_loop(enrol_dataloader)
    test_dict = compute_embedding_loop(test_dataloader)

    if "score_norm" in params:
        train_dict = compute_embedding_loop(train_dataloader)

    logger.info("Computing EER and minDCF...")

    with open(veri_file_path, encoding="utf-8") as f:
        veri_test = [line.rstrip() for line in f]

    positive_scores, negative_scores = get_verification_scores(veri_test)

    eer, eer_th = EER(
        torch.tensor(positive_scores),
        torch.tensor(negative_scores),
    )
    logger.info("EER(%%)=%.4f", eer * 100.0)
    logger.info("EER threshold=%.6f", float(eer_th))

    min_dcf, min_dcf_th = minDCF(
        torch.tensor(positive_scores),
        torch.tensor(negative_scores),
    )
    logger.info("minDCF=%.6f", float(min_dcf))
    logger.info("minDCF threshold=%.6f", float(min_dcf_th))