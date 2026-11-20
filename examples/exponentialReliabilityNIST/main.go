// example from the NIST/SEMATECH
// e-Handbook of Statistical Methods, section 8.4.6:
// https://itl.nist.gov/div898/handbook/apr/section4/apr46.htm
//
// Failure times are exponential with failure rate λ (MTBF = 1/λ).
// Prior: λ ~ Gamma(a=2, b=1400) (shape a, rate b in hours).
// New test: r = 2 failures in T = 1909 total test hours.
// Exact posterior: λ ~ Gamma(4, 3309). NIST reports these MTBF bounds:
//
//	90% lower bound 495 h, 80% lower bound 600 h, median 901 h,
//	10% lower bound 1897 h.
package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	r := 2
	T := 1909.0

	lambda := &variable.Variable{
		Name:           "lambda",
		Value:          0.001,
		Prior:          distributions.Gamma{Alpha: 2, Beta: 1400},
		ProposalScale:  0.6,
		Multiplicative: true,
	}

	likelihood := func(params map[string]float64) float64 {
		l := params["lambda"]
		if l <= 0 {
			return math.Inf(-1)
		}
		e := distributions.Exponential{Lambda: l}
		ll := 0.0
		for i := 0; i < r; i++ {
			ll += e.LogProb(T / float64(r))
		}
		return ll
	}

	m := &model.Model{
		Variables: []*variable.Variable{lambda},
		LogLik:    likelihood,
	}
	result, err := m.Run(model.Options{Iterations: 400000, BurnIn: 20000})
	if err != nil {
		panic(err)
	}

	mtbf := make([]float64, len(result.Draws[0][0]))
	sumLambda := 0.0
	for i, s := range result.Draws[0][0] {
		sumLambda += s
		mtbf[i] = 1 / s
	}
	sort.Float64s(mtbf)
	q := func(p float64) float64 { return mtbf[int(p*float64(len(mtbf)-1))] }

	fmt.Printf("Posterior mean of λ ≈ %.6f (exact 4/3309 = %.6f)\n", sumLambda/float64(len(result.Draws[0][0])), 4.0/3309.0)
	fmt.Printf("MTBF 90%% lower bound ≈ %.0f h (NIST: 495)\n", q(0.10))
	fmt.Printf("MTBF 80%% lower bound ≈ %.0f h (NIST: 600)\n", q(0.20))
	fmt.Printf("MTBF median          ≈ %.0f h (NIST: 901)\n", q(0.50))
	fmt.Printf("MTBF 10%% lower bound ≈ %.0f h (NIST: 1897)\n", q(0.90))
	io.PrintDiagnostics(result)
}
