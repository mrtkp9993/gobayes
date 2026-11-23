// Bernoulli-beta inference corresponding to Kruschke's DBDA2E example
// Jags-Ydich-Xnom1subj-MbernBeta (z15N50.csv).
package main

import (
	"fmt"
	"math"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	y := make([]float64, 50)
	for i := 0; i < 15; i++ {
		y[i] = 1
	}

	m := &model.Model{
		Variables: []*variable.Variable{{
			Name:          "theta",
			Value:         0.3,
			Prior:         distributions.Uniform{A: 0, B: 1}, // Beta(1,1)
			ProposalScale: 0.1,
		}},
		LogLik: func(params map[string]float64) float64 {
			d := distributions.Bernoulli{P: params["theta"]}
			var logProb float64
			for _, observation := range y {
				logProb += d.LogProb(observation)
			}
			return logProb
		},
	}

	result, err := m.Run(model.Options{
		Iterations: 2500, BurnIn: 1000, Chains: 4, Seed: 20261123,
	})
	if err != nil {
		panic(err)
	}
	var sum float64
	count := 0
	for _, chain := range result.Draws {
		for _, x := range chain[0] {
			sum += x
			count++
		}
	}
	fmt.Printf("Posterior mean of theta ≈ %.6f\n", sum/float64(count))
	fmt.Printf("Analytic Beta(16,36): mean %.6f; SD %.6f\n",
		16.0/52, math.Sqrt(16.0*36/(52*52*53)))
	bayesio.PrintDiagnostics(result)
}
