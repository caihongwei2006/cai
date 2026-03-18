package jsrt

import (
	"context"
	"fmt"
	"sync"
	"time"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
	"github.com/dop251/goja"
)

// Runtime is an embedded JavaScript runtime powered by goja (pure Go V8-lite).
// Supports simple JS/TS execution without external subprocess.
// For full TS with npm dependencies, use TSBunEngine instead.
type Runtime struct {
	pool       sync.Pool
	classifier *envelope.Classifier
}

// New creates a new embedded JS runtime.
func New() *Runtime {
	return &Runtime{
		pool: sync.Pool{
			New: func() any {
				vm := goja.New()
				vm.SetFieldNameMapper(goja.TagFieldNameMapper("json", true))
				return vm
			},
		},
		classifier: envelope.DefaultClassifier(),
	}
}

func (r *Runtime) Type() cai.EngineType { return cai.EngineNodeJS }

// Execute runs a JavaScript snippet in an isolated VM instance.
// The VM is pooled for reuse across invocations.
func (r *Runtime) Execute(ctx context.Context, script string, env map[string]string) (*cai.ExecutionEnvelope, error) {
	vm := r.pool.Get().(*goja.Runtime)
	defer func() {
		clearVM(vm)
		r.pool.Put(vm)
	}()

	// Inject environment as global `env` object
	envObj := vm.NewObject()
	for k, v := range env {
		envObj.Set(k, v)
	}
	vm.Set("env", envObj)

	// Capture console.log output
	var output string
	vm.Set("__output", func(call goja.FunctionCall) goja.Value {
		for _, arg := range call.Arguments {
			output += arg.String()
		}
		output += "\n"
		return goja.Undefined()
	})

	consolePatch := `var console = { log: __output, info: __output, warn: __output, error: __output };`

	// Context-aware execution with timeout
	done := make(chan struct{})
	var result goja.Value
	var runErr error

	go func() {
		defer close(done)
		result, runErr = vm.RunString(consolePatch + "\n" + script)
	}()

	select {
	case <-ctx.Done():
		vm.Interrupt("context cancelled")
		<-done
		env := r.classifier.Classify(124, output, "timeout: context deadline exceeded")
		return &env, nil
	case <-done:
	}

	if runErr != nil {
		env := r.classifier.Classify(1, output, runErr.Error())
		return &env, nil
	}

	if result != nil && !goja.IsUndefined(result) && !goja.IsNull(result) {
		output += fmt.Sprint(result.Export())
	}

	env2 := r.classifier.Classify(0, output, "")
	return &env2, nil
}

func (r *Runtime) Close() error { return nil }

func clearVM(vm *goja.Runtime) {
	vm.ClearInterrupt()
	vm.Set("env", goja.Undefined())
	vm.Set("__output", goja.Undefined())
}

// ExecuteWithTimeout runs JS with an explicit timeout.
func (r *Runtime) ExecuteWithTimeout(script string, env map[string]string, timeout time.Duration) (*cai.ExecutionEnvelope, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return r.Execute(ctx, script, env)
}
