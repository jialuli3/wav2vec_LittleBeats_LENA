#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc phoneme recognition system with MyST corpus.
This recipe is adapted from original SpeechBrain project from Librispeech corpus.
The system employs wav2vec as its encoder. Decoding is performed with ctc greedy decoder.
To run this recipe, do the following:
> python train_with_wav2vec_phos_MyST.py hparams/train_with_wav2vec_phos_Libri_MyST.yaml
The neural network is trained on CTC likelihood target and phoneme units
are used as basic recognition tokens. 
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.modules.wav2vec2(wavs)
        x = self.modules.enc(feats)
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)
        
        p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )

        if stage != sb.Stage.TRAIN:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.hparams, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = loss_ctc

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ndim(utt_seq)
                for utt_seq in predicted_tokens
            ]                  

            target_words = [wrd.split(" ") for wrd in batch.pho]
            self.wer_metric.append(ids, predicted_words, target_words)
        
        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0

        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with self.no_sync():
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                if not self.hparams.freeze_wav2vec:
                    self.scaler.unscale_(self.wav2vec_optimizer)
                self.scaler.unscale_(self.model_optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.wav2vec_optimizer)
                    self.scaler.step(self.model_optimizer)
                self.scaler.update()
                self.optimizer_step += 1
        else:
            with self.no_sync():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.wav2vec_optimizer.step()
                    self.model_optimizer.step()
                self.wav2vec_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                self.optimizer_step += 1

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.out = self.hparams.output_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=stage_stats, min_keys=["WER"], num_to_keep=2
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            #self.out.write_stats(self.hparams.wer_file, self.hparams.tg_prefix)
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
            self.wer_metric.write_wer_sigtest(self.hparams.out_json_file)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)

    def reset_lr(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.hparams.lr_annealing_model.hyperparam_value = self.hparams.lr
        self.hparams.lr_annealing_wav2vec.hyperparam_value = self.hparams.lr_wav2vec

        sb.nnet.schedulers.update_learning_rate(
            self.model_optimizer, self.hparams.lr
        )
        sb.nnet.schedulers.update_learning_rate(
            self.wav2vec_optimizer, self.hparams.lr_wav2vec
        )

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"], replacements={"data_root": data_folder},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("phonemes")
    @sb.utils.data_pipeline.provides(
        "pho", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(pho):
        yield pho
        #char_list = list(wrd)
        char_list = pho.split(" ")
        char_list = [x.replace("ɑʊ","aʊ") for x in char_list] # get rid of one uncommon phoneme in Providence corpus
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "pho", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

    asr_brain.tokenizer = label_encoder

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    asr_brain.hparams.wer_file = hparams["wer_file"]
    asr_brain.evaluate(
        test_datasets, 
        min_key="WER", 
        test_loader_kwargs=hparams["test_dataloader_opts"]
    )
