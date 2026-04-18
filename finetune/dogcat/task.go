package dogcat

import "github.com/gomlx/gomlx/pkg/core/dtypes"

const MinimumImageSize = 75

const (
	PreGeneratedTrainFileName      = "train_data.bin"
	PreGeneratedTrainPairFileName  = "train_pair_data.bin"
	PreGeneratedTrainEvalFileName  = "train_eval_data.bin"
	PreGeneratedValidationFileName = "validation_eval_data.bin"
)

// PreprocessingConfiguration holds various parameters on how to transform the input images.
type PreprocessingConfiguration struct {
	// DataDir, where downloaded and generated data is stored.
	DataDir string

	// DType of the images when converted to Tensor.
	DType dtypes.DType

	// BatchSize for training and evaluation batches.
	BatchSize, EvalBatchSize int

	// ModelImageSize is use for height and width of the generated images.
	ModelImageSize int

	// YieldImagePairs if to yield an extra input with the paired image: same image, different random augmentation.
	// Only applies for Train dataset.
	YieldImagePairs bool

	// NumFolds for cross-validation.
	NumFolds int

	// Folds to use for train and validation.
	TrainFolds, ValidationFolds []int

	// FoldsSeed used when randomizing the folds assignment, so it can be done deterministically.
	FoldsSeed int32

	// AngleStdDev for angle perturbation of the image. Only active if > 0.
	AngleStdDev float64

	// FlipRandomly the image, for data augmentation. Only active if true.
	FlipRandomly bool

	// ForceOriginal will make CreateDatasets not use the pre-generated augmented datasets, even if
	// they are present.
	ForceOriginal bool

	// UseParallelism when using Dataset.
	UseParallelism bool

	// BufferSize used for datasets.ParallelDataset, to cache intermediary batches. This value is used
	// for each dataset.
	BufferSize int

	// NumSamples is the maximum number of samples the model is allowed to see. If set to -1
	// model can see all samples.
	NumSamples int
}

var (
	DefaultConfig = &PreprocessingConfiguration{
		DType:           dtypes.Float32,
		BatchSize:       16,
		EvalBatchSize:   100, // Faster evaluation with larger batches.
		ModelImageSize:  MinimumImageSize,
		NumFolds:        5,
		TrainFolds:      []int{0, 1, 2, 3},
		ValidationFolds: []int{4},
		FoldsSeed:       0,
		UseParallelism:  true,
		BufferSize:      32,
		NumSamples:      -1,
	} // DType used for model.
)
