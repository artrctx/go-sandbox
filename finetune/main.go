package main

import (
	"finetune/dogcat"
	"flag"
	"os"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"
)

var (
	flagDataDir    = flag.String("data", "./finetune/dataset/PetImages", "Directory to cache downloaded dataset and save checkpoints.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate trained model on test data in the end.")

	// Pre-Generation parameters:
	flagPreGenerate = flag.Bool("pre", false, "Pre-generate preprocessed image data to speed up training.")
	flagPreGenEpoch = flag.Int("pregen_epochs", 40, "Number of epochs to pre-generate for the training data. Each epoch will take ~310Mb")
)

// https://go.dev/wiki/AI
// https://github.com/gomlx/gomlx
// https://arxiv.org/abs/2502.01225
// https://github.com/gomlx/gomlx/tree/main/examples/dogsvscats
func main() {
	ctx := dogcat.CreateDefultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	ps := check1(commandline.ParseContextSettings(ctx, *settings))

	if err := exceptions.TryCatch[error](func() {
		if *flagPreGenerate {
			preGenerate(ctx, *flagDataDir)
		} else {

		}

	}); err != nil {
		klog.Fatal(err)
	}
}

func preGenerate(ctx *context.Context, dataDir string) {
	flagDataDir := fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(flagDataDir) {
		check(os.MkdirAll(flagDataDir, 0777))
	}
}

// check reports and exits on error.
func check(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}

// check1 reports and exits on error. Otherwise returns the value passed.
func check1[T any](v T, err error) T {
	check(err)
	return v
}
