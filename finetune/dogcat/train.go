package dogcat

import (
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gomlx/ui/gonb/plotly"
)

// CreateDefaultContext sets the context with default hyperparameters to use with TrainModel.
func CreateDefultContext() *context.Context {
	ctx := context.New()
	ctx.ResetRNGState()
	ctx.SetParams(map[string]any{
		// Model type to use
		"model":           "cnn",
		"num_checkpoints": 3,
		"train_steps":     2000,

		// batch_size for training.
		"batch_size": DefaultConfig.BatchSize,

		// eval_batch_size can be larger than training, it's more efficient.
		"eval_batch_size": DefaultConfig.EvalBatchSize,

		// Debug parameters.
		"nan_logger": false, // Trigger nan error as soon as it happens -- expensive, but helps debugging.

		// Image augmentation parameters
		"augmentation_angle_stddev":   20.0,  // Standard deviation of noise used to rotate the image. Disabled if --augment=false.
		"augmentation_random_flips":   true,  // Randomly flip images horizontally.
		"augmentation_force_original": false, // Force reading from original images instead of pre-generated.

		// "plots" trigger generating intermediary eval data for plotting, and if running in GoNB, to actually
		// draw the plot with Plotly.
		//
		// From the command-line, an easy way to monitor the metrics being generated during the training of a model
		// is using the gomlx_checkpoints tool:
		//
		//	$ gomlx_checkpoints --metrics --metrics_labels --metrics_types=accuracy  --metrics_names='E(Tra)/#loss,E(Val)/#loss' --loop=3s "<checkpoint_path>"
		plotly.ParamPlots: true,

		// "normalization" is overridden by "fnn_normalization" and "cnn_normalization", if they are set.
		layers.ParamNormalization: "batch",

		optimizers.ParamOptimizer:       "adamw",
		optimizers.ParamLearningRate:    1e-4,
		optimizers.ParamAdamEpsilon:     1e-7,
		optimizers.ParamAdamDType:       "",
		cosineschedule.ParamPeriodSteps: 0,
		activations.ParamActivation:     "",
		layers.ParamDropoutRate:         0.1,
		regularizers.ParamL2:            0.0,
		regularizers.ParamL1:            0.0,

		// FNN network parameters:
		fnn.ParamNumHiddenLayers: 3,
		fnn.ParamNumHiddenNodes:  128,
		fnn.ParamResidual:        true,
		fnn.ParamNormalization:   "",   // Set to "none" for no normalization, otherwise it falls back to layers.ParamNormalization.
		fnn.ParamDropoutRate:     -1.0, // Set to 0.0 for no dropout, otherwise it falls back to layers.ParamDropoutRate.

		// KAN network parameters:
		kan.ParamNumControlPoints:   10, // Number of control points
		kan.ParamNumHiddenNodes:     64,
		kan.ParamNumHiddenLayers:    4,
		kan.ParamBSplineDegree:      2,
		kan.ParamBSplineMagnitudeL1: 1e-5,
		kan.ParamBSplineMagnitudeL2: 0.0,
		kan.ParamDiscrete:           false,
		kan.ParamDiscreteSoftness:   0.1,

		// CNN
		"cnn_num_layers":      5.0,
		"cnn_dropout_rate":    -1.0,
		"cnn_embeddings_size": 128,

		// BYOL (Build Your Own Latent) model configuration ("model": "byol")
		"byol_pretrain":            false, // Pre-train BYOL model, unsupervised.
		"byol_finetune":            false, // Finetune BYOL model. If set to false, only the linear model on top is trained.
		"byol_hidden_nodes":        4096,  // Number of nodes in the hidden layer.
		"byol_projection_nodes":    256,   // Number of nodes (dimension) in the projection to the target regularizing model.
		"byol_target_update_ratio": 0.99,  // Moving average update weight to the "target" sub-model for BYOL model.
		"byol_regularization_rate": 1.0,   // BYOL regularization loss rate, a simple multiplier.
		"byol_inception":           false, // Instead of using a CNN model with BYOL, uses InceptionV3.
		"byol_reg_len1":            0.01,  // BYOL regularize projections to length 1.

		// InceptionV3 model configuration ("model": "inception")
		"inception_pretrained": true, // Whether to use the pre-trained weights to transfer learn
		"inception_finetuning": true, // Whether to fine-tune the inception model
	})

	return ctx
}
