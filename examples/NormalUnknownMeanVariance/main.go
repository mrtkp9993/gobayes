// Translation of: https://sites.google.com/view/omori-stat/english/software/winbugs/normal1?authuser=0
package main

import (
	"fmt"
	"math"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	x := []float64{
		182, 201, 221, 234, 237, 251, 261, 266, 267, 273,
		286, 291, 292, 296, 296, 296, 326, 352, 359, 365,
	}
	n := len(x)
	mu := &variable.Variable{
		Name:          "mu",
		Value:         0.0,
		Prior:         distributions.Normal{Mu: 0, Sigma: 1000},
		ProposalScale: 5.0,
	}
	invSigma2 := &variable.Variable{
		Name:           "invSigma2",
		Value:          1.0,
		Prior:          distributions.Gamma{Alpha: 0.0001, Beta: 0.0001},
		ProposalScale:  0.5,
		Multiplicative: true,
	}
	likelihood := func(params map[string]float64) float64 {
		muVal := params["mu"]
		tau := params["invSigma2"]
		if tau <= 0 {
			return math.Inf(-1)
		}
		ll := 0.0
		ll += float64(n) * 0.5 * (math.Log(tau) - math.Log(2*math.Pi))
		for _, t := range x {
			ll += -0.5 * tau * math.Pow(t-muVal, 2)
		}
		return ll
	}
	model := &model.Model{
		Variables: []*variable.Variable{mu, invSigma2},
		LogLik:    likelihood,
	}
	samples, err := model.RunMCMC(100000, 10000)
	if err != nil {
		panic(err)
	}
	sumMu := 0.0
	sumSigma := 0.0
	for _, s := range samples {
		sumMu += s["mu"]
		tau := s["invSigma2"]
		sigma2 := 1.0 / tau
		sumSigma += math.Sqrt(sigma2)
	}
	muMean := sumMu / float64(len(samples))
	sigmaMean := sumSigma / float64(len(samples))
	fmt.Printf("Posterior mean of μ ≈ %.4f, σ ≈ %.4f\n", muMean, sigmaMean)
}
