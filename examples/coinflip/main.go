package main

import (
	"fmt"
	"math"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/internal/exampleutil"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	// Observed data: 8 heads, 4 tails
	heads := 8
	tails := 4

	theta := &variable.Variable{
		Name:          "theta",
		Value:         0.5,
		Prior:         distributions.Uniform{A: 0.0, B: 1.0},
		ProposalScale: 0.1,
	}

	likelihood := func(params map[string]float64) float64 {
		t := params["theta"]
		if t <= 0 || t >= 1 {
			return math.Inf(-1)
		}
		return float64(heads)*math.Log(t) + float64(tails)*math.Log(1.0-t)
	}

	m := &model.Model{
		Variables: []*variable.Variable{theta},
		LogLik:    likelihood,
	}

	result, err := m.Run(model.Options{Iterations: 10000, BurnIn: 1000})
	if err != nil {
		panic(err)
	}

	sum := 0.0
	for _, s := range result.Draws[0][0] {
		sum += s
	}
	mean := sum / float64(len(result.Draws[0][0]))
	fmt.Printf("Posterior mean of θ ≈ %.4f\n", mean)
	exampleutil.PrintDiagnostics(result)
}
