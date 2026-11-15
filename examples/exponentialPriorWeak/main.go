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
	mtbf50 := 600.0
	b := math.Ln2 * mtbf50
	r := 2
	T := 1909.0

	prior := distributions.Exponential{Lambda: b}

	ps := make([]float64, 200000)
	for i := range ps {
		ps[i] = prior.Sample()
	}
	sort.Float64s(ps)
	fmt.Printf("Prior median MTBF from Exponential.Sample ≈ %.0f h (elicited: 600)\n", 1/ps[len(ps)/2])

	lambda := &variable.Variable{
		Name:           "lambda",
		Value:          0.001,
		Prior:          prior,
		ProposalScale:  0.6,
		Multiplicative: true,
	}
	likelihood := func(params map[string]float64) float64 {
		l := params["lambda"]
		if l <= 0 {
			return math.Inf(-1)
		}
		return float64(r)*math.Log(l) - l*T
	}
	m := &model.Model{
		Variables: []*variable.Variable{lambda},
		LogLik:    likelihood,
	}
	samples, err := m.RunMCMC(400000, 20000)
	if err != nil {
		panic(err)
	}
	sum, sumSq := 0.0, 0.0
	for _, s := range samples {
		sum += s["lambda"]
		sumSq += s["lambda"] * s["lambda"]
	}
	n := float64(len(samples))
	mean := sum / n
	sd := math.Sqrt(sumSq/n - mean*mean)
	bPost := b + T
	fmt.Printf("Posterior mean of λ ≈ %.7f (exact 3/%.2f = %.7f)\n", mean, bPost, 3/bPost)
	fmt.Printf("Posterior sd of λ   ≈ %.7f (exact √3/%.2f = %.7f)\n", sd, bPost, math.Sqrt(3)/bPost)
}
