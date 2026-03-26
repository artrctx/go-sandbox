package main

import (
	"fmt"
	"num_recog/mnist"
)

// https://medium.com/@Kiana-Jafari/digit-recognition-building-a-deep-neural-network-from-fmtscratch-132bd574564a
func main() {
	dataset, err := mnist.ReadDataset("num_recog/dataset/train-labels-idx1-ubyte", "num_recog/dataset/train-images-idx3-ubyte")
	if err != nil {
		panic(err)
	}
	fmt.Printf("N %v W %v H %v", dataset.N, dataset.W, dataset.H)
}
