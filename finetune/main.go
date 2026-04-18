package main

import (
	"k8s.io/klog/v2"
)

// https://go.dev/wiki/AI
// https://github.com/gomlx/gomlx
// https://arxiv.org/abs/2502.01225
// https://github.com/gomlx/gomlx/tree/main/examples/dogsvscats
func main() {
	klog.InitFlags(nil)
	klog.Infoln("THIS IS INFO")
	klog.Errorln("THIS IS ERROR")
}
