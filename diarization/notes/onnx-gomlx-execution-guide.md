# ONNX-GoMLX Execution Guide

This guide explains the lifecycle behind this pattern:

```go
exec, err := model.NewExec(
	backend,
	store,
	func(scope *model.Scope, inputs []*model.Node) []*model.Node {
		return ogModel.CallGraph(
			scope,
			inputs[0].Graph(),
			inputMap,
			outputNames...,
		)
	},
)
```

The key distinction is that the callback **builds a symbolic graph**. It does not run inference. `model.NewExec` later specializes, compiles, caches, and executes that graph for concrete input tensors.

This guide was checked against `onnx-gomlx v0.5.5`, `gomlx v0.28.8`, `compute v0.1.6`, `go-xla v0.4.5`, and the optional `compute-onnx v0.1.5`. Consult the release notes when using materially different versions.

## End-to-End Lifecycle

The normal lifecycle is:

1. Parse the ONNX file into an `onnx.Model`.
2. Create a GoMLX backend and model store.
3. Import ONNX initializers, such as weights and biases, into the store's scope.
4. Create a reusable executor with `model.NewExec`.
5. Pass concrete tensors or Go values to the executor.
6. Reuse the executor for subsequent calls.

In outline:

```go
ogModel, err := parser.ParseFile(modelPath)
if err != nil {
	return fmt.Errorf("parse ONNX model: %w", err)
}

backend := backends.New()
store := model.NewStore()

// ONNX initializers become GoMLX variables. Do this once during setup, before
// constructing or calling the executor.
if err := ogModel.VariablesToScope(store.RootScope()); err != nil {
	return fmt.Errorf("load ONNX initializers: %w", err)
}

inputNames, _ := ogModel.Inputs()
outputNames, _ := ogModel.Outputs()
if len(inputNames) == 0 {
	return errors.New("ONNX model has no runtime inputs")
}

exec, err := model.NewExec(
	backend,
	store,
	func(scope *model.Scope, inputs []*model.Node) []*model.Node {
		inputMap := make(map[string]*model.Node, len(inputNames))
		for index, name := range inputNames {
			inputMap[name] = inputs[index]
		}

		return ogModel.CallGraph(
			scope,
			inputs[0].Graph(),
			inputMap,
			outputNames...,
		)
	},
)
if err != nil {
	return fmt.Errorf("create ONNX executor: %w", err)
}

outputs, err := exec.Call(concreteInputValues...)
if err != nil {
	return fmt.Errorf("execute ONNX model: %w", err)
}

// Keep exec for repeated calls. Its owner should call exec.Finalize() during
// shutdown, after the final call has completed. The owner should also close the
// parsed model with ogModel.Close().
```

The concrete values passed to `exec.Call` depend on the application. They may already be GoMLX tensors or supported Go values that GoMLX converts to tensors.

## What Parsing Does

`parser.ParseFile`, `parser.Parse`, and `parser.ParseReader` read the ONNX protobuf and return an `onnx.Model`.

The parsed model contains:

- Graph inputs and outputs.
- Initializers, including trained weights and constant tensors.
- ONNX nodes and their attributes.
- Shape and type information available in the model.
- Referenced subgraphs, where applicable.

Parsing does not compile the model for a backend and does not run inference.

Use `ogModel.Inputs()` and `ogModel.Outputs()` to obtain declared names and shapes. Using the returned input order as the executor argument order avoids relying on assumptions about names or protobuf ordering elsewhere in the application.

## Loading Weights

`ogModel.VariablesToScope(scope)` copies ONNX initializers into GoMLX variables under the model's ONNX scope.

Run it once during initialization, before `CallGraph` is used. Reimporting initializers for every request is unnecessary and can overwrite or duplicate setup work, depending on how the surrounding store is managed.

The `scope` passed to `CallGraph` must resolve the same variables. In practice, pass the store whose root scope received the initializers to `model.NewExec`.

## What `model.NewExec` Does

`model.NewExec` creates a reusable model executor. Its three arguments have separate responsibilities:

### `backend`

The backend compiles and executes the generated graph. Backend choice determines the available device/runtime behavior and may affect supported operations and concurrency guarantees.

## ONNX Importer Versus Execution Backend

`onnx-gomlx` and `compute-onnx` occupy different layers despite their similar names:

| Package | Responsibility |
| --- | --- |
| `github.com/gomlx/onnx-gomlx` | Parses an existing `.onnx` model and translates it into a symbolic GoMLX graph |
| `github.com/gomlx/compute-onnx` | Optionally compiles and executes GoMLX graphs through ONNX Runtime |

Using an ONNX model does not require the ONNX Runtime backend. With XLA CPU, the execution path is:

```text
.onnx file
  -> onnx-gomlx parser
  -> GoMLX graph
  -> XLA/PJRT CPU backend
  -> execution
```

With the pure-Go backend, it is:

```text
.onnx file
  -> onnx-gomlx parser
  -> GoMLX graph
  -> pure-Go compute backend
  -> execution
```

Neither path uses `compute-onnx` or ONNX Runtime. Import `github.com/gomlx/compute-onnx` only when ONNX Runtime is intentionally selected as the GoMLX execution backend.

### Selecting a backend in code

There is no backend-independent `compute.NewCPU()` or `compute.NewGPU()`. CPU and GPU selection is specific to the chosen runtime. Use either the compute registry or a backend's direct constructor.

The registry accepts the same configuration format as `GOMLX_BACKEND`:

```go
import (
	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
)

backend, err := compute.NewWithConfig("xla:cpu")
if err != nil {
	return fmt.Errorf("create XLA CPU backend: %w", err)
}
```

`github.com/gomlx/gomlx/backends/default` registers the XLA and pure-Go backends on supported platforms. The ONNX Runtime backend must be separately registered by importing `github.com/gomlx/compute-onnx`; GoMLX's default package only does this automatically for a Linux AMD64 binary built with the `onnx` tag.

Examples of registered configurations are:

```go
compute.NewWithConfig("xla:cpu")
compute.NewWithConfig("xla:cuda")
compute.NewWithConfig("xla:rocm")
compute.NewWithConfig("xla:tpu")
compute.NewWithConfig("go")
compute.NewWithConfig("go:parallelism=8,ops_parallel")
compute.NewWithConfig("onnx:cpu")
compute.NewWithConfig("onnx:cuda,log=2")
```

Only configurations whose backend packages were compiled into and registered with the program are available. In `compute v0.1.6`, `compute.NewWithConfig` panics when the requested backend name is not registered, although constructor and initialization failures are returned as errors.

For an application that deliberately requires one runtime, a direct constructor is clearer and avoids registry selection.

### XLA/PJRT

Create an explicit XLA CPU backend with:

```go
import (
	"github.com/gomlx/go-xla/compute/xla"
	_ "github.com/gomlx/go-xla/compute/xla/autoinstall"
)

backend, err := xla.New("cpu")
```

Other XLA plugin selections include:

```go
backend, err := xla.New("cuda")
backend, err := xla.New("rocm")
backend, err := xla.New("tpu")
```

Options follow the plugin name, separated by commas:

```go
backend, err := xla.New(
	"cuda,preallocate=false,allocator=cuda_async,visible_devices=0",
)
```

Calling `xla.New("")` selects the first available plugin in `cuda`, `rocm`, then `cpu` preference order. Use `xla.New("cpu")` when CPU execution must be deterministic rather than inferred from the machine.

XLA does not require ONNX Runtime, but it does require the selected PJRT plugin. Importing `github.com/gomlx/go-xla/compute/xla/autoinstall` enables automatic plugin installation. Without it, the plugin must already be installed and discoverable. Set `GOMLX_NO_AUTO_INSTALL=1` or call `xla.EnableAutoInstall(false)` to prohibit automatic downloads.

On Apple Silicon, `xla.New("cpu")` selects XLA CPU. These pinned packages do not provide a generic Apple Metal GPU backend.

### Pure Go

The pure-Go backend requires no ONNX Runtime, PJRT plugin, or native ML runtime:

```go
import "github.com/gomlx/compute/gobackend"

backend, err := gobackend.New("")
```

It can also be configured explicitly:

```go
backend, err := gobackend.New(
	"parallelism=8,ops_parallel,dependency_order",
)
```

The pure-Go backend prioritizes portability. It may support fewer operations and generally provides lower performance than XLA, so verify that it can compile the imported model.

### ONNX Runtime backend

Use `compute-onnx` only when ONNX Runtime is the intended execution engine:

```go
import onnxbackend "github.com/gomlx/compute-onnx"

backend, err := onnxbackend.New("cpu")
```

Examples include:

```go
backend, err := onnxbackend.New("cuda,log=2")
backend, err := onnxbackend.New("migraphx")
```

This backend needs an ONNX Runtime shared library. It can locate an existing installation or automatically install a supported binary unless automatic installation is disabled. This requirement is unique to choosing the ONNX Runtime backend; it does not apply when `onnx-gomlx` is paired with XLA or pure Go.

### Backend ownership

All constructors above return a `compute.Backend` suitable for `model.NewExec`. A component that creates the backend owns it unless ownership is explicitly transferred. After all executors, stores, and tensors using it have been finalized, call `backend.Finalize()`. Do not finalize a backend injected from and shared with another component.

### `store`

The store owns model variables and their values, including imported ONNX initializers. It also connects symbolic variables referenced while building the graph to concrete values used during execution.

### Graph callback

The callback translates the parsed ONNX model into GoMLX graph nodes for the current graph specialization.

```go
func(scope *model.Scope, inputs []*model.Node) []*model.Node
```

- `scope` resolves the variables used by the model.
- `inputs` contains symbolic input nodes created for this graph.
- The returned nodes define the executor's ordered outputs.

GoMLX supports multiple callback signatures and canonicalizes them internally. The slice-based form is useful for a model with a dynamic number of inputs or outputs.

## What `CallGraph` Does

`ogModel.CallGraph` walks the parsed ONNX graph and creates equivalent symbolic GoMLX operations:

```go
ogModel.CallGraph(scope, graph, inputMap, outputNames...)
```

Its arguments are:

### `scope`

Used to resolve weights and other ONNX initializers imported into the store.

### `graph`

The GoMLX graph receiving the translated operations. In the slice callback, the graph is obtained from one of the current input nodes:

```go
inputs[0].Graph()
```

All nodes supplied in `inputMap` must belong to this graph. A node from an earlier callback invocation or another executor cannot be reused.

### `inputMap`

Maps each ONNX graph input name to the corresponding symbolic GoMLX input node.

For a two-input model:

```go
inputNames := []string{"input_ids", "attention_mask"}

func(scope *model.Scope, inputs []*model.Node) []*model.Node {
	inputMap := map[string]*model.Node{
		inputNames[0]: inputs[0],
		inputNames[1]: inputs[1],
	}

	return ogModel.CallGraph(
		scope,
		inputs[0].Graph(),
		inputMap,
		outputNames...,
	)
}
```

Build this map inside the callback. Do not capture a map containing `*model.Node` values created outside the callback because those nodes belong to a different graph.

Capturing immutable input name strings is safe. The callback combines those names with the fresh nodes supplied by GoMLX.

### `outputNames`

Selects outputs by ONNX value name and determines their returned order. It can select declared model outputs or named intermediate values supported by the importer.

When no output names are passed, `CallGraph` returns the model's registered outputs.

## Symbolic Nodes Versus Concrete Values

There are two separate layers:

| Layer | Values | Purpose |
| --- | --- | --- |
| Graph construction | `*model.Node` | Describes operations and dependencies symbolically |
| Execution | tensors or supported Go values | Supplies data and receives computed results |

The callback only handles symbolic nodes. Tensor data is supplied later through the executor's call API.

This is why `CallGraph` can be invoked more than once even though the ONNX model was parsed only once: each invocation translates the model into a new graph specialization.

## Shape Specialization and Caching

The executor specializes graphs for concrete input shapes and types.

On the first call for a shape/type combination, GoMLX may:

1. Create symbolic inputs matching the concrete values.
2. Invoke the graph callback.
3. Translate the ONNX operations through `CallGraph`.
4. Compile the resulting graph for the backend.
5. Cache the executable.
6. Run it.

A later call with the same specialization can reuse the cached executable. A new input shape or type can trigger another graph build and compilation.

By default, `Exec` permits up to 32 cached graph specializations. A call requiring another specialization then returns an error. `exec.SetMaxCache(n)` changes the limit; `-1` makes it unlimited, which should only be used with bounded input variability.

Cache size counts distinct input signatures, not goroutines. A signature includes the complete ordered set of input shapes and dtypes. For example:

```text
float32 [1, 768]  -> specialization 1
float32 [8, 768]  -> specialization 2
float64 [8, 768]  -> specialization 3
```

An ONNX model has one input schema, but dynamic dimensions can permit several concrete signatures. A fully static input such as `float32[1, 3, 224, 224]` normally needs only one specialization. An input such as `float32[batch, 3, 224, 224]` can create one specialization per concrete batch size unless that axis is configured as dynamic.

If five goroutines simultaneously call a cold executor with the same signature, one goroutine builds the graph and the others wait for that compilation. They then use the same cached graph; GoMLX does not create one graph per goroutine. If all five calls have different signatures, GoMLX can compile five different graphs concurrently.

`SetMaxCache(5)` only sets an upper bound. It does not preallocate five graphs or create a worker pool.

### Cache eviction

The executor cache is not LRU or LFU. It does not track which graph is most or least frequently used, and it does not automatically evict an entry.

When the cache is full:

- Calls matching existing signatures continue to reuse their graphs.
- A call requiring a new signature returns an error.
- Lowering the configured maximum does not remove existing entries.
- `exec.Finalize()` clears the entire cache and permanently closes that executor.

There is no public API to remove one specialization. If automatic eviction is required, manage separate executors in an application-level cache and finalize an evicted executor only after its active calls finish. Prefer dynamic axes, padding, or a bounded set of shape buckets before adding that complexity.

Consequences:

- The first inference is normally slower than steady-state inference.
- Highly variable shapes can create multiple cached executables and compilation pauses.
- Prewarming expected shapes can move compilation cost out of a latency-sensitive request path.
- Inputs must remain compatible with the ONNX operations even when the executor accepts a new shape specialization.

`exec.Compile(inputShapes...)` builds and caches a specialization without running inference. Use it to precompile known shapes. `exec.WithDynamicAxes(...)` can allow selected dimensions, commonly batch size, to vary without recompilation when both the model and backend support the resulting dynamic graph.

For example, an input with dynamic batch size and a static width can be configured with:

```go
exec.WithDynamicAxes([]string{"batch", ""})
```

Each dynamic axis needs a non-empty name. Each static axis uses an empty string, and each axis-name slice must match the corresponding input rank.

### Warmup requirements

Warm each expected signature, not each goroutine. When all ONNX dimensions are static, the model's declared shapes can be used for compile-only warmup:

```go
_, inputShapes := ogModel.Inputs()
if _, err := exec.Compile(inputShapes...); err != nil {
	return fmt.Errorf("precompile ONNX model: %w", err)
}
```

`Compile` warms graph construction and compilation but does not execute the graph. To also warm input transfers, backend execution, and kernels, call the executor once with representative values:

```go
if _, err := exec.Call(dummyInputs...); err != nil {
	return fmt.Errorf("warm ONNX model: %w", err)
}
```

There is no universal default input. Dummy values must have the correct count, dtype, rank, and dimensions, and should be semantically safe for the model. If a dynamic dimension has no known representative size, full warmup is not possible; let the first real call compile lazily. `model.NewExec` itself does not require concrete input values or shapes.

## Safer Generic Callback

For a reusable wrapper, validate the input count before indexing and create graph-bound state inside the callback:

```go
inputNames := slices.Clone(configuredInputNames)
outputNames := slices.Clone(configuredOutputNames)

exec, err := model.NewExec(
	backend,
	store,
	func(scope *model.Scope, inputs []*model.Node) []*model.Node {
		if len(inputs) != len(inputNames) {
			panic(fmt.Sprintf(
				"ONNX model expects %d inputs, received %d",
				len(inputNames),
				len(inputs),
			))
		}

		inputMap := make(map[string]*model.Node, len(inputNames))
		for index, name := range inputNames {
			inputMap[name] = inputs[index]
		}

		return ogModel.CallGraph(
			scope,
			inputs[0].Graph(),
			inputMap,
			outputNames...,
		)
	},
)
if err != nil {
	return fmt.Errorf("create ONNX executor: %w", err)
}
```

Notes:

- Validate that `inputNames` is non-empty before creating the executor; otherwise `inputs[0]` will panic.
- Validate configured input and output names during application startup when possible.
- Prefer names returned by `ogModel.Inputs()` and `ogModel.Outputs()` when using all declared inputs and outputs.
- Clone configuration slices if another part of the application could mutate them.
- A panic in graph construction is appropriate only if the surrounding API treats an invalid model configuration as a programmer/startup error. Otherwise, validate before entering the callback and return an ordinary setup error.

### Callback panic behavior

`model.NewExec` creates the executor but does not normally invoke the callback. The callback runs on the first `Call` or `Compile` for each uncached specialization. Cache hits do not invoke it again.

When the callback runs through `Exec`, GoMLX recovers a panic raised by the callback or by `CallGraph` and converts it to an error. Failed graph construction is not added to the cache.

```go
outputs, err := exec.Call(inputs...)
if err != nil {
	// A graph-building panic is reported here as an error.
}
```

The error-returning APIs do not terminate the program:

| API | Result of a callback panic |
| --- | --- |
| `exec.Call(...)` | Returns an error |
| `exec.Compile(...)` | Returns an error |
| `exec.MustCall(...)` | Re-panics after receiving the recovered error |

An unrecovered panic from `MustCall` terminates the program. Server code should generally use `Call` and handle the returned error. Calling `ogModel.CallGraph` directly, outside an `Exec` graph-building invocation, does not provide the executor's recovery boundary.

## Concurrency

Treat three concurrency concerns independently.

### Graph construction

GoMLX can build specializations for different shapes concurrently. The callback should therefore avoid mutating captured maps, slices, counters, or other shared state.

Building `inputMap` locally is safe. Captured input and output name slices should be immutable after executor creation.

Some ONNX control-flow translation paths temporarily manipulate model graph mappings and are not safe to build concurrently. If the model contains ONNX control-flow subgraphs, serialize first-time compilation or precompile expected shapes sequentially.

### Variables

Inference weights should remain read-only after initialization. Concurrent mutation of variables while executions are reading them is unsafe unless the application provides explicit synchronization.

### Backend execution

Do not assume every backend permits concurrent calls through one executor. The common conservative choices are:

1. Serialize calls to a shared executor with a mutex.
2. Use one executor per worker when parallel execution is required.
3. Confirm the selected backend's concurrency contract before allowing unrestricted shared execution.

Executor compilation caching and executor thread safety are different properties. A reusable cache does not by itself guarantee that simultaneous calls are safe.

For calls with the same signature, concurrent compilation is deduplicated, but execution is not serialized: after compilation, goroutines can reach the same backend executable concurrently. The generic `compute.Executable` interface does not promise that every backend supports this. Use a mutex around `Call`, or isolate executors as required, unless the selected backend documents concurrent `Execute` support.

## Initialization and Request Boundaries

Keep model setup outside the request path:

```text
Process initialization
  parse ONNX model
  create backend and store
  import initializers
  create executor
  optionally prewarm expected shapes

Per request
  convert request data to tensors
  call executor
  convert output tensors to application values
```

Do not parse the ONNX file, import weights, or construct an executor for every inference unless isolation requirements explicitly demand it.

These resources have separate responsibilities:

- `exec.Finalize()` clears compiled graphs and releases their backend executable resources. It does not close the parsed ONNX model or release variables in the store. The executor must not be used afterward.
- `ogModel.Close()` closes readers and file handles used for ONNX external tensor data. It does not clear executor graphs or release store variables.
- `store.Finalize()` releases the store's variable tensors. Call it only when the component owns the store rather than receiving a shared store.
- `backend.Finalize()` releases the backend runtime and device resources. Call it last, after every executor, store, and tensor using that backend is finished, and only when the component owns the backend.

The component that owns these resources should shut them down only after calls have stopped:

```go
exec.Finalize()
store.Finalize()
closeErr := ogModel.Close()
backend.Finalize()
if closeErr != nil {
	return fmt.Errorf("close ONNX model: %w", closeErr)
}
```

Because the executor callback captures `ogModel`, keep the parsed model open for the executor's entire lifetime. A cache miss can invoke the callback again to construct another specialization.

If a runner implements `io.Closer` and owns all four resources, `Close` is the appropriate place to finalize them. It must not race an active `Call`. One pattern is to hold a read lock for each call and a write lock during shutdown:

```go
type Runner struct {
	mu       sync.RWMutex
	closed   bool
	closeErr error
	exec     *model.Exec
	store    *model.Store
	model    onnx.Model
	backend  compute.Backend
}

func (r *Runner) Call(args ...any) ([]*tensors.Tensor, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if r.closed {
		return nil, errors.New("ONNX runner is closed")
	}
	return r.exec.Call(args...)
}

func (r *Runner) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return r.closeErr
	}
	r.closed = true

	r.exec.Finalize()
	r.store.Finalize()
	r.closeErr = r.model.Close()
	r.backend.Finalize()
	return r.closeErr
}
```

This permits concurrent calls but makes `Close` wait for all of them. It does not change the backend concurrency caveat described above. If the wrapper does not own an injected executor, store, model, or backend, it must not close that shared resource; the creator remains responsible for its lifetime. Production code may also preserve errors from every fallible cleanup operation rather than returning only the model close error.

## Error Handling

Wrap errors at lifecycle boundaries so failures identify their stage:

```go
ogModel, err := parser.ParseFile(modelPath)
if err != nil {
	return fmt.Errorf("parse ONNX model %q: %w", modelPath, err)
}

exec, err := model.NewExec(/* ... */)
if err != nil {
	return fmt.Errorf("create executor for ONNX model %q: %w", modelPath, err)
}
```

Execution errors should include non-sensitive model and input metadata, such as model identity, input names, shapes, and element types. Avoid logging complete tensors because they can be large or contain sensitive data.

Graph translation errors often indicate one of the following:

- An unsupported ONNX operation or attribute.
- An input name that does not match the ONNX graph.
- A requested output name that is unavailable.
- Incompatible input shapes or element types.
- A missing initializer or incorrect scope/store setup.
- Nodes from different GoMLX graphs being combined.

## Practical Checklist

- Parse the ONNX model once.
- Import initializers into the executor's store hierarchy once.
- Keep ONNX input names in the same order as concrete executor arguments.
- Build `inputMap` inside the graph callback.
- Use only nodes from the callback's current `inputs` slice.
- Treat `outputNames` order as part of the wrapper's API contract.
- Expect first-call compilation for each shape/type specialization.
- Account for the default 32-entry specialization cache or set an intentional limit.
- Remember that cache size counts signatures, not goroutines.
- Do not expect automatic LRU or LFU eviction.
- Prewarm stable production shapes when startup time permits.
- Warm each expected signature once, not once per worker.
- Consider `WithDynamicAxes` for dimensions that genuinely need to vary.
- Keep inference variables immutable.
- Serialize compilation for models with control-flow subgraphs.
- Verify backend concurrency guarantees before sharing one executor across goroutines.
- Finalize an owned backend after its executors, stores, and tensors are no longer in use.
- Include input names, shapes, and types in execution error context.

## Interpretation of the Original Snippet

Line by line, the original code means:

```go
exec, err := model.NewExec(
```

Create a reusable GoMLX model executor.

```go
	backend,
```

Use this backend to compile and run graph specializations.

```go
	store,
```

Use this variable store, which should already contain the ONNX initializers.

```go
	func(scope *model.Scope, inputs []*model.Node) []*model.Node {
```

Define how to build a symbolic graph for the current concrete input specialization.

```go
		return ogModel.CallGraph(
```

Translate the parsed ONNX graph into GoMLX operations.

```go
			scope,
```

Resolve model variables from the executor's store hierarchy.

```go
			inputs[0].Graph(),
```

Build operations in the graph containing the current symbolic inputs.

```go
			inputMap,
```

Bind ONNX input names to symbolic input nodes. This map should be constructed from the current `inputs` inside the callback.

```go
			outputNames...,
```

Select and order the ONNX values exposed as executor outputs.

```go
		)
	},
)
```

Finish the graph-building callback and create the reusable executor. No inference has occurred yet.

## Versioned References

- [ONNX-GoMLX README and inference example](https://github.com/gomlx/onnx-gomlx/blob/v0.5.5/README.md)
- [`onnx.Model` interface](https://github.com/gomlx/onnx-gomlx/blob/v0.5.5/onnx/onnx.go)
- [`CallGraph` implementation](https://github.com/gomlx/onnx-gomlx/blob/v0.5.5/internal/onnxgomlx/graph.go)
- [`model.NewExec` and call helpers](https://github.com/gomlx/gomlx/blob/v0.28.8/ml/model/execaliases.go)
- [`model.Exec` variable integration](https://github.com/gomlx/gomlx/blob/v0.28.8/ml/model/exec.go)
- [Graph executor caching and concurrency](https://github.com/gomlx/gomlx/blob/v0.28.8/core/graph/exec.go)
- [Compute backend registration and configuration](https://github.com/gomlx/compute/blob/v0.1.6/compute.go)
- [Pure-Go backend constructor and options](https://github.com/gomlx/compute/blob/v0.1.6/internal/gobackend/gobackend.go)
- [XLA/PJRT backend constructor and options](https://github.com/gomlx/go-xla/blob/v0.4.5/compute/xla/xla.go)
- [ONNX Runtime backend](https://github.com/gomlx/compute-onnx/blob/v0.1.5/backend.go)
