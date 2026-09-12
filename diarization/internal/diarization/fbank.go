package diarization

//https://docs.nvidia.com/nemo/speech/nightly/asr/api.html
//https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
//https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html#preprocessor-configuration
// https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/preprocessing/features.py

/*
SortformerEncLabelModel
        │
        ▼
model.preprocessor
        │
        ▼
AudioToMelSpectrogramPreprocessor
        │
        ▼
self.featurizer
        │
        ▼
FilterbankFeatures
        │
        ▼
FilterbankFeatures.forward()


[FilterbankFeatures]
sample_rate               = 16000
win_length                = 400
hop_length                = 160
n_fft                     = 512
window                    = Tensor(shape=(400,))
normalize                 = NA
preemph                   = 0.97
nfilt                     = 128
log                       = True
log_zero_guard_type       = add
log_zero_guard_value      = 5.960464477539063e-08
dither                    = 1e-05
pad_to                    = 16
frame_splicing            = 1
exact_pad                 = False
pad_value                 = 0
mag_power                 = 2.0
nb_augmentation_prob      = 0.0
*/
