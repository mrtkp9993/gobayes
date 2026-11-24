package main

import (
	"flag"
	"fmt"
	"slices"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	iterations := flag.Int("iterations", 20000, "saved draws per chain")
	warmup := flag.Int("warmup", 10000, "warmup sweeps per chain")
	chains := flag.Int("chains", 4, "number of chains")
	seed := flag.Uint64("seed", 42, "MCMC seed (data seed is fixed separately)")
	flag.Parse()
	data := struct {
		Seed uint64
		Y    []float64
	}{
		Seed: 20261124,
		Y: []float64{
			2, 0, 7, 4, 11, 1, 8, 4, 2, 2,
			0, 7, 1, 1, 4, 3, 9, 4, 0, 2,
			4, 6, 11, 9, 1, 1, 0, 0, 6, 4,
			5, 3, 0, 6, 6, 4, 2, 1, 7, 2,
			0, 6, 5, 7, 2, 6, 4, 4, 1, 0,
			3, 5, 2, 2, 3, 1, 10, 7, 8, 8,
			4, 2, 5, 6, 4, 8, 2, 3, 4, 5,
			1, 1, 3, 2, 0, 1, 3, 8, 0, 4,
			4, 0, 2, 2, 3, 0, 10, 3, 1, 7,
			3, 1, 7, 0, 6, 1, 0, 1, 11, 4,
		},
	}
	counts := make([]int, 1+int(slices.Max(data.Y)))
	for _, y := range data.Y {
		counts[int(y)]++
	}
	m := model.Model{
		Variables: []*variable.Variable{
			{Name: "mu", Value: 4, Prior: distributions.Gamma{Alpha: 0.01, Beta: 0.01}, ProposalScale: 0.2, Multiplicative: true},
			{Name: "alpha", Value: 2, Prior: distributions.Gamma{Alpha: 0.01, Beta: 0.01}, ProposalScale: 0.3, Multiplicative: true}},
		LogLik: func(params map[string]float64) float64 {
			mu, alpha := params["mu"], params["alpha"]
			d := distributions.NegativeBinomial{R: alpha, P: alpha / (alpha + mu)}
			lp := 0.0
			for y, count := range counts {
				if count != 0 {
					lp += float64(count) * d.LogProb(float64(y))
				}
			}
			return lp
		},
	}
	result, err := m.Run(model.Options{Iterations: *iterations, BurnIn: *warmup, Chains: *chains,
		Seed: *seed, Adapt: true, Componentwise: true, TargetAccept: 0.44})
	if err != nil {
		panic(err)
	}
	monitored := *result
	monitored.Names = []string{"mu", "alpha", "p", "variance"}
	monitored.Draws = make([][][]float64, len(result.Draws))
	for c, chain := range result.Draws {
		monitored.Draws[c] = make([][]float64, 4)
		for j := range monitored.Names {
			monitored.Draws[c][j] = make([]float64, len(chain[0]))
		}
		for i, mu := range chain[0] {
			alpha := chain[1][i]
			p := alpha / (alpha + mu)
			for j, x := range []float64{mu, alpha, p, mu / p} {
				monitored.Draws[c][j][i] = x
			}
		}
	}
	fmt.Printf("Synthetic N=%d, data seed=%d, true mu=4 alpha=2\n", len(data.Y), data.Seed)
	for j, name := range monitored.Names {
		var sum float64
		count := 0
		for _, chain := range monitored.Draws {
			for _, x := range chain[j] {
				sum += x
				count++
			}
		}
		fmt.Printf("Posterior mean of %s = %.6f\n", name, sum/float64(count))
	}
	bayesio.PrintDiagnostics(&monitored)
}
