package main

import (
	"fmt"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
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
		return (distributions.Binomial{N: heads + tails, P: params["theta"]}).LogProb(float64(heads))
	}

	m := &model.Model{
		Variables: []*variable.Variable{theta},
		LogLik:    likelihood,
	}

	result, err := m.Run(model.Options{Iterations: 10000, BurnIn: 1000, Chains: 4})
	if err != nil {
		panic(err)
	}

	sum := 0.0
	for _, s := range result.Draws[0][0] {
		sum += s
	}
	mean := sum / float64(len(result.Draws[0][0]))
	fmt.Printf("Posterior mean of θ ≈ %.4f\n", mean)
	bayesio.PrintDiagnostics(result)
}
