"""
This snippet is adapted from the original SpeechBrain codebase.
This lobe enables the integration of fairseq pretrained wav2vec models.
Reference: https://arxiv.org/abs/2006.11477
Reference: https://arxiv.org/abs/1904.05862
FairSeq >= 1.0.0 needs to be installed: https://fairseq.readthedocs.io/en/latest/

Original Authors
 * Titouan Parcollet 2021
 * Salima Mdhaffar 2021
 
Modified by
 * Jialu Li 2023
"""

import torch
import logging
import torch.nn.functional as F
from torch import nn
from speechbrain.utils.data_utils import download_file
from speechbrain.dataio.dataio import length_to_mask

# We check if fairseq is installed.
try:
    import fairseq
except ImportError:
    MSG = "Please install Fairseq to use pretrained wav2vec\n"
    MSG += "E.G. run: pip install fairseq"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class FairseqWav2Vec2(nn.Module):
    """This lobe enables the integration of fairseq pretrained wav2vec2.0 models.
    Source paper: https://arxiv.org/abs/2006.11477
    FairSeq >= 1.0.0 needs to be installed:
    https://fairseq.readthedocs.io/en/latest/
    The model can be used as a fixed features extractor or can be finetuned. It
    will download automatically the model if a url is given (e.g FairSeq
    repository from GitHub).
    Arguments
    ---------
    save_path : str
        Path and filename of the downloaded model.
    input_norm : bool (default: None)
        If True, a layer_norm (affine) will be applied to the input waveform.
        By default, it is extracted from the checkpoint of the downloaded model
        in order to match the pretraining conditions. However, if this information
        is not given in the checkpoint, it has to be given manually.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.
    dropout : float (default: None)
        If different from None (0.0 to 1.0), it will override the given fairseq
        dropout rates. This is useful if the wav2vec2 model has been trained
        without dropout and one wants to reactivate it for downstream task
        fine-tuning (better performance observed).
    encoder_dropout : float (default: None)
        If different from None (0.0 to 1.0), it will override the given fairseq
        encoder_layerdrop rates. It has certain probability to dropout random number
        of layer features.
    output_all_hiddens: bool (default: False)
        If True, output the features from all 12 transformer layers.
        If False, output the features from only the last transformer layer.
    tgt_layer: int or list of int (default: None)
        If not None, output the features of the front-end CNN or specified transformer layer(s).
        (0-indexed. 0 - CNN front-end layer, 1-12 transformer layers).
        For extracting front-end CNN features, specify it as "CNN".
        For single layer, specify it as an int.
        For multiple layers, specify it as a list of int.
    include_CNN_layer: bool (default: False)
        This should be used when output_all_hiddens==True.
        If True, output the features from front-end CNN layer as well as all 12 transformer layers.
    """

    def __init__(
        self,
        pretrained_path,
        save_path,
        input_norm=None,
        output_norm=True,
        freeze=True,
        pretrain=True,
        dropout=None,
        encoder_dropout = None, 
        output_all_hiddens = True,
        tgt_layer = None, 
        include_CNN_layer = False,
    ):
        super().__init__()

        # Download the pretrained wav2vec2 model. It can be local or online.
        download_file(pretrained_path, save_path)
        # During pretraining dropout might be set to 0. However, we might want
        # to apply dropout when fine-tuning on a downstream task. Hence we need
        # to modify the fairseq cfg to activate dropout (if requested).
        overrides={}
        if encoder_dropout is not None:
            overrides = {
                "model": {
                    "encoder_layerdrop": encoder_dropout,
                }
            }
        if not freeze:
            if dropout is not None and encoder_dropout is not None:
                overrides = {
                    "model": {
                        "dropout": dropout,
                        "encoder_layerdrop": encoder_dropout,
                        "dropout_input": dropout,
                        "attention_dropout": dropout,
                    }
                }
            elif dropout is not None:
                overrides = {
                    "model": {
                        "dropout": dropout,
                        "dropout_input": dropout,
                        "attention_dropout": dropout,
                    }
                }     
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [save_path], arg_overrides=overrides
        )

        # wav2vec pretrained models may need the input waveform to be normalized
        # Hence, we check if the model has be trained with or without it.
        # If the information isn't contained in the checkpoint IT HAS TO BE GIVEN
        # BY THE USER.
        if input_norm is None:
            if hasattr(cfg["task"], "normalize"):
                self.normalize = cfg["task"].normalize
            elif hasattr(cfg, "normalize"):
                self.normalize = cfg.normalize
            else:
                self.normalize = False
        else:
            self.normalize = input_norm

        model = model[0]
        self.model = model
        self.freeze = freeze
        self.output_norm = output_norm

        if self.freeze:
            self.model.eval()
            # Freeze parameters
            for param in model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in model.parameters():
                param.requires_grad = True

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

        # Following the fairseq implementation of downstream training,
        # we remove some modules that are unnecessary.
        self.remove_pretraining_modules()
        self.output_all_hiddens = output_all_hiddens
        self.tgt_layer = tgt_layer
        self.include_CNN_layer=include_CNN_layer
        if not self.output_all_hiddens:
            logger.info(
                f"include_CNN_layer is not used when output_all_hidden is False"
            )
        if self.output_all_hiddens:
            self.tgt_layer==None
            logger.warning(
                f"Set tgt_layer to None when output_all_hiddens is True"
            )

    def forward(self, wav):
        """Takes an input waveform of shape (Batch, Time) and return its corresponding wav2vec encoding.
        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Extracts the wav2vect embeddings
        wav: torch tensor 
        Retruning output dimension as # of Layers x Batch x Time x Dimension
        """
        # We normalize the input signal if needed.
        if self.normalize:
            wav = F.layer_norm(wav, wav.shape)

        out = self.model.extract_features(wav, padding_mask=None, mask=False)
        # Extract wav2vec output
        if isinstance(self.tgt_layer, int):
            features = out['layer_results'][self.tgt_layer][0].transpose(0, 1)
        elif isinstance(self.tgt_layer, list):
            features = []        
            for i in self.tgt_layer:
                curr_feature = out['layer_results'][i][0].transpose(0, 1)
                features.append(curr_feature)
            features = torch.stack(features)
        elif self.output_all_hiddens:
            features = self.aggregate_features(out, include_CNN_layer=self.include_CNN_layer) # 13, B, T, D
        else: # output last layer only
            features = out['x']

        out=features   
        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out
    
    def aggregate_features(self, out):
        features = []
        for i in range(len(out['layer_results'])):
            curr_feature = out['layer_results'][i][0].transpose(0,1)
            features.append(curr_feature)
        features = torch.stack(features)
        return features
        
    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)

    def _load_sb_pretrained_w2v2_parameters(self, path):
        """Loads the parameter of a w2v2 model pretrained with SpeechBrain and the
        HuggingFaceWav2Vec2Pretrain Object. It is necessary to perform a custom
        loading because HuggingFace adds a level to the checkpoint when storing
        the model breaking the compatibility between HuggingFaceWav2Vec2Pretrain
        and HuggingFaceWav2Vec2.
        In practice a typical HuggingFaceWav2Vec2 checkpoint for a given parameter
        would be: model.conv.weight.data while for HuggingFaceWav2Vec2Pretrain it
        is: model.wav2vec2.weight.data (wav2vec2 must be removed before loading).
        """
        modified_state_dict = {}
        orig_state_dict = torch.load(path, map_location="cpu")

        # We remove the .wav2vec2 in the state dict.
        for key, params in orig_state_dict.items():
            if "model." in key:
                save_key = key.replace("model.", "")
                modified_state_dict[save_key] = params

        incompatible_keys = self.model.load_state_dict(
            modified_state_dict, strict=False
        )

        for missing_key in incompatible_keys.missing_keys:
            logger.warning(
                f"During parameter transfer to {self.model} loading from "
                + f"{path}, the transferred parameters did not have "
                + f"parameters for the key: {missing_key}"
            )

        for unexpected_key in incompatible_keys.unexpected_keys:
            logger.warning(
                f"The param with the key: {unexpected_key} is discarded as it "
                + "is useless for wav2vec 2.0 finetuning."
            )

    def remove_pretraining_modules(self):
        """ Remove uneeded modules. Inspired by the same fairseq function."""

        self.model.quantizer = None
        self.model.project_q = None
        self.model.target_glu = None
        self.model.final_proj = None


class FairseqWav2Vec1(nn.Module):
    """This lobes enables the integration of fairseq pretrained wav2vec1.0 models.

    Arguments
    ---------
    pretrained_path : str
        Path of the pretrained wav2vec1 model. It can be a url or a local path.
    save_path : str
        Path and filename of the downloaded model.
    output_norm : bool (default: True)
        If True, a layer_norm (affine) will be applied to the output obtained
        from the wav2vec model.
    freeze : bool (default: True)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    pretrain : bool (default: True)
        If True, the model is pretrained with the specified source.
        If False, the randomly-initialized model is instantiated.

    Example
    -------
    >>> inputs = torch.rand([10, 600])
    >>> model_url = ""
    >>> save_path = "models_checkpoints/wav2vec.pt"
    >>> model = FairseqWav2Vec1(model_url, save_path)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 100, 512])
    """

    def __init__(
        self,
        pretrained_path,
        save_path,
        output_norm=True,
        freeze=True,
        pretrain=True,
    ):
        super().__init__()
        self.freeze = freeze
        self.output_norm = output_norm

        # Download the pretrained wav2vec1 model. It can be local or online.
        download_file(pretrained_path, save_path)

        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [pretrained_path]
        )

        self.model = model
        self.model = self.model[0]
        if self.freeze:
            self.model.eval()

        # Randomly initialized layers if pretrain is False
        if not (pretrain):
            self.reset_layer(self.model)

    def forward(self, wav):
        """Takes an input waveform and return its corresponding wav2vec encoding.

        Arguments
        ---------
        wav : torch.Tensor (signal)
            A batch of audio signals to transform to features.
        """

        # If we freeze, we simply remove all grads and features from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(wav).detach()

        return self.extract_features(wav)

    def extract_features(self, wav):
        """Extracts the wav2vect embeddings"""

        out = self.model.feature_extractor(wav)
        out = self.model.feature_aggregator(out).squeeze(0)
        out = out.transpose(2, 1)

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, out.shape)

        return out

    def reset_layer(self, model):
        """Reinitializes the parameters of the network"""
        if hasattr(model, "reset_parameters"):
            model.reset_parameters()
        for child_layer in model.children():
            if model != child_layer:
                self.reset_layer(child_layer)
