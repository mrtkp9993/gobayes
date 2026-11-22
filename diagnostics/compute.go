package diagnostics

import (
	"errors"
	"math"
)

func Compute(chains [][]float64) (ParameterDiagnostics, error) {
	var result ParameterDiagnostics
	if len(chains) == 0 || len(chains[0]) < 4 {
		return result, errors.New("at least one chain with four draws is required")
	}
	n, m := len(chains[0]), len(chains)
	means := make([]float64, m)
	vars := make([]float64, m)
	var grand, w float64
	for c, xs := range chains {
		if len(xs) != n {
			return result, errors.New("chains must have equal lengths")
		}
		for _, x := range xs {
			if math.IsNaN(x) || math.IsInf(x, 0) {
				return result, errors.New("draws must be finite")
			}
			means[c] += x / float64(n)
		}
		grand += means[c] / float64(m)
		for _, x := range xs {
			d := x - means[c]
			vars[c] += d * d / float64(n-1)
		}
		w += vars[c] / float64(m)
	}
	if w == 0 {
		return result, errors.New("ESS, MCSE and R-hat are unavailable because all chains have constant draws")
	}
	if w < 0 || math.IsInf(w, 0) || math.IsNaN(w) {
		return result, errors.New("within-chain variance must be finite and positive")
	}
	var between float64
	if m > 1 {
		for _, mean := range means {
			d := mean - grand
			between += float64(n) * d * d / float64(m-1)
		}
	}
	varPlus := (float64(n-1)*w + between) / float64(n)
	if math.IsInf(varPlus, 0) || math.IsNaN(varPlus) {
		return result, errors.New("variance estimate is not finite")
	}
	if m > 1 {
		rh := rankNormalizedSplitRHat(chains)
		result.RHat = &rh
	}
	maxLag := min(n-1, 1000)
	result.Autocorrelation = make([]float64, maxLag+1)
	result.Autocorrelation[0] = 1
	for lag := 1; lag <= maxLag; lag++ {
		var cov float64
		for c, xs := range chains {
			var sum float64
			for i := 0; i < n-lag; i++ {
				sum += (xs[i] - means[c]) * (xs[i+lag] - means[c])
			}
			cov += sum / float64(n*m)
		}
		result.Autocorrelation[lag] = 1 - (w-cov)/varPlus
	}
	tau := -1.0
	for lag := 0; lag+1 <= maxLag; lag += 2 {
		pair := result.Autocorrelation[lag] + result.Autocorrelation[lag+1]
		if pair <= 0 {
			break
		}
		tau += 2 * pair
	}
	total := float64(n) * float64(m)
	if tau < 1/total {
		tau = 1 / total
	}
	result.ESS = math.Min(total, total/tau)
	result.MCSEMean = math.Sqrt(varPlus / result.ESS)
	return result, nil
}
