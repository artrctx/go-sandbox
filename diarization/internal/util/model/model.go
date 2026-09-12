package model

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
)

type Model struct {
	model     onnx.Model
	backend   compute.Backend
	store     *model.Store
	exec      *model.Exec
	terminate chan struct{}
}

type config struct {
	backend string
}

type Option func(*config)

var DefaultCachePath string = func() string {
	cwd, err := os.Getwd()
	if err != nil {
		panic(err)
	}
	return cwd + "/models"
}()

func WithCPU() Option {
	return func(m *config) {
		m.backend = "xla:cpu"
	}
}

func WithGPU() Option {
	return func(m *config) {
		m.backend = "xla:gpu"
	}
}

func New(path string, opts ...Option) (*Model, error) {
	if path == "" {
		return nil, errors.New("remote model path is required")
	}

	ext := filepath.Ext(path)
	if ext != ".onnx" {
		return nil, errors.New("model needs to be onnx file")
	}

	// ------ Optional Config Set
	cfg := &config{
		backend: "go",
	}

	for _, opt := range opts {
		opt(cfg)
	}

	m := &Model{}
	// ------ Initializing Onnx Model
	backend, err := compute.NewWithConfig(cfg.backend)
	if err != nil {
		return nil, err
	}
	m.backend = backend

	m.store = model.NewStore()
	m.model, err = parser.ParseFile(path)
	if err != nil {
		m.Close()
		return nil, err
	}

	if err := m.model.VariablesToScope(m.store.RootScope()); err != nil {
		m.Close()
		return nil, err
	}

	inputNames, _ := m.model.Inputs()
	outputNames, _ := m.model.Outputs()
	if len(inputNames) == 0 {
		m.Close()
		return nil, errors.New("ONNX model has no runtime inputs")
	}

	m.exec, err = model.NewExec(backend, m.store, func(scope *model.Scope, inputs []*model.Node) []*model.Node {
		if len(inputs) == len(inputNames) {
			panic(fmt.Sprintf(
				"ONNX model expect %d inputs (%s) but got %d",
				len(inputNames),
				strings.Join(inputNames, ","),
				len(inputs)))
		}

		inputMap := make(map[string]*model.Node, len(inputNames))
		for i, n := range inputNames {
			inputMap[n] = inputs[i]
		}

		return m.model.CallGraph(scope, inputs[0].Graph(), inputMap, outputNames...)
	})

	if err != nil {
		m.Close()
		return nil, err
	}

	return m, nil
}

func (m *Model) WarmUp(shapes ...shapes.Shape) error {
	_, err := m.exec.Compile(shapes...)
	return err
}

func (m *Model) Run(args ...any) ([]*tensors.Tensor, error) {
	return m.exec.Call(args...)
}

func (m *Model) Close() (err error) {
	if m.terminate != nil {
		close(m.terminate)
	}
	if m.exec != nil {
		m.exec.Finalize()
	}
	if m.store != nil {
		m.store.Finalize()
	}
	if m.model != nil {
		err = m.model.Close()
	}
	if m.backend != nil {
		m.backend.Finalize()
	}
	return
}
