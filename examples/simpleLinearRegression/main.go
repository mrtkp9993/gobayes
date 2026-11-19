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
	xs := []float64{-2, -1, 0, 1, 2}
	ys := []float64{-5.1, -2.8, -0.2, 2.1, 4.9}
	sigma := 0.5
	alpha := &variable.Variable{
		Name:          "alpha",
		Value:         0.0,
		Prior:         distributions.Normal{Mu: 0, Sigma: 10},
		ProposalScale: 0.2,
	}
	beta := &variable.Variable{
		Name:          "beta",
		Value:         0.0,
		Prior:         distributions.Normal{Mu: 0, Sigma: 10},
		ProposalScale: 0.2,
	}
	// Log likelihood for linear regression
	likelihood := func(params map[string]float64) float64 {
		a := params["alpha"]
		b := params["beta"]
		ll := 0.0
		for i := range xs {
			mu := a + b*xs[i]
			// Gaussian likelihood: log N(ys[i] | mu, sigma)
			ll += -0.5*math.Log(2*math.Pi*sigma*sigma) - math.Pow(ys[i]-mu, 2)/(2*sigma*sigma)
		}
		return ll
	}
	m := &model.Model{
		Variables: []*variable.Variable{alpha, beta},
		LogLik:    likelihood,
	}
	result, err := m.Run(model.Options{Iterations: 20000, BurnIn: 2000})
	if err != nil {
		panic(err)
	}
	// Compute posterior means
	sumAlpha := 0.0
	sumBeta := 0.0
	for i, alpha := range result.Draws[0][0] {
		sumAlpha += alpha
		sumBeta += result.Draws[0][1][i]
	}
	meanAlpha := sumAlpha / float64(len(result.Draws[0][0]))
	meanBeta := sumBeta / float64(len(result.Draws[0][0]))
	fmt.Printf("Posterior mean of α ≈ %.4f, β ≈ %.4f\n", meanAlpha, meanBeta)
	exampleutil.PrintDiagnostics(result)
}
