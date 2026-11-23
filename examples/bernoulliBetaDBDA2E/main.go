// Bernoulli-beta inference corresponding to Kruschke's DBDA2E example
// Jags-Ydich-Xnom1subj-MbernBeta (z15N50.csv).
package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/mrtkp9993/gobayes/distributions"
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
	var samples []float64
	var sum float64
	for _, chain := range result.Draws {
		for _, x := range chain[0] {
			samples = append(samples, x)
			sum += x
		}
	}
	mean := sum / float64(len(samples))
	var sumSquares float64
	for _, x := range samples {
		sumSquares += (x - mean) * (x - mean)
	}
	sort.Float64s(samples)
	quantile := func(p float64) float64 {
		position := p * float64(len(samples)-1)
		i := int(position)
		return samples[i] + (position-float64(i))*(samples[i+1]-samples[i])
	}
	diagnostics, err := result.ParameterDiagnostics("theta")
	if err != nil {
		panic(err)
	}
	fmt.Println("DBDA2E Bernoulli-beta: 15 successes / 50 trials; prior Beta(1,1)")
	fmt.Printf("Seed: %d; chains: %d; saved draws: %d\n", result.Seed, len(result.Draws), len(samples))
	fmt.Printf("Posterior mean: %.6f; SD: %.6f; 95%% equal-tail interval: [%.6f, %.6f]\n",
		mean, math.Sqrt(sumSquares/float64(len(samples)-1)), quantile(0.025), quantile(0.975))
	fmt.Printf("ESS: %.1f; MCSE(mean): %.6f; R-hat: %.6f\n",
		diagnostics.ESS, diagnostics.MCSEMean, *diagnostics.RHat)
	fmt.Printf("Analytic Beta(16,36): mean %.6f; SD %.6f\n",
		16.0/52, math.Sqrt(16.0*36/(52*52*53)))
}
