// Categorical dice inference with a uniform Dirichlet prior.
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
	data := struct{ Counts [6]int }{Counts: [6]int{5, 8, 9, 10, 12, 16}}
	names := [6]string{"w1", "w2", "w3", "w4", "w5", "w6"}

	m := &model.Model{}
	for _, name := range names {
		m.Variables = append(m.Variables, &variable.Variable{
			Name: name, Value: 1, Prior: distributions.Gamma{Alpha: 1, Beta: 1},
			ProposalScale: 0.5, Multiplicative: true,
		})
	}
	m.LogLik = func(params map[string]float64) float64 {
		p := make([]float64, len(names))
		var total float64
		for k, name := range names {
			p[k] = params[name]
			total += p[k]
		}
		for k := range p {
			p[k] /= total
		}
		d := distributions.Categorical{P: p}
		var logProb float64
		for k, count := range data.Counts {
			logProb += float64(count) * d.LogProb(float64(k))
		}
		return logProb
	}

	result, err := m.Run(model.Options{
		Iterations: 20000, BurnIn: 10000, Chains: 4, Seed: 42,
		Adapt: true, Componentwise: true, TargetAccept: 0.44,
	})
	if err != nil {
		panic(err)
	}

	monitored := *result
	monitored.Names = make([]string, len(names))
	monitored.Draws = make([][][]float64, len(result.Draws))
	for k := range names {
		monitored.Names[k] = fmt.Sprintf("p[%d]", k+1)
	}
	for c, chain := range result.Draws {
		monitored.Draws[c] = make([][]float64, len(names))
		for k := range names {
			monitored.Draws[c][k] = make([]float64, len(chain[0]))
		}
		for i := range chain[0] {
			var total float64
			for k := range names {
				total += chain[k][i]
			}
			for k := range names {
				monitored.Draws[c][k][i] = chain[k][i] / total
			}
		}
	}

	var concentration float64
	for _, count := range data.Counts {
		concentration += float64(count + 1)
	}
	fmt.Println("Posterior: Dirichlet(6,9,10,11,13,17)")
	fmt.Println("Face  Go mean   Go SD     Analytic mean  Analytic SD")
	for k := range names {
		var sum, squaredDeviations float64
		n := 0
		for _, chain := range monitored.Draws {
			for _, x := range chain[k] {
				sum += x
				n++
			}
		}
		mean := sum / float64(n)
		for _, chain := range monitored.Draws {
			for _, x := range chain[k] {
				squaredDeviations += (x - mean) * (x - mean)
			}
		}
		sd := math.Sqrt(squaredDeviations / float64(n-1))
		a := float64(data.Counts[k] + 1)
		analyticMean := a / concentration
		analyticSD := math.Sqrt(a * (concentration - a) / (concentration * concentration * (concentration + 1)))
		fmt.Printf("%d     %.6f  %.6f  %.6f       %.6f\n",
			k+1, mean, sd, analyticMean, analyticSD)
	}
	bayesio.PrintDiagnostics(&monitored)
}
