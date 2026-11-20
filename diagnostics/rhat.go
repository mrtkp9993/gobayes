package diagnostics

import (
	"cmp"
	"math"
	"slices"
)

func rankNormalizedSplitRHat(chains [][]float64) float64 {
	half := len(chains[0]) / 2
	draws := make([]float64, 0, 2*half*len(chains))
	for _, chain := range chains {
		draws = append(draws, chain[:half]...)
		draws = append(draws, chain[len(chain)-half:]...)
	}

	scores, median := rankNormalize(draws)
	rhat := splitRHat(scores, half)
	for i, draw := range draws {
		draws[i] = math.Abs(draw - median)
	}
	foldedScores, _ := rankNormalize(draws)
	if folded := splitRHat(foldedScores, half); folded > rhat {
		return folded
	}
	return rhat
}

func rankNormalize(draws []float64) (scores []float64, median float64) {
	order := make([]int, len(draws))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(i, j int) int {
		return cmp.Compare(draws[i], draws[j])
	})
	middle := len(order) / 2
	median = 0.5*draws[order[middle-1]] + 0.5*draws[order[middle]]
	scores = make([]float64, len(draws))
	for first := 0; first < len(order); {
		end := first + 1
		for end < len(order) && draws[order[end]] == draws[order[first]] {
			end++
		}
		rank := (float64(first+1) + float64(end)) / 2
		p := (rank - 0.375) / (float64(len(order)) + 0.25)
		score := -math.Sqrt2 * math.Erfcinv(2*p)
		for _, index := range order[first:end] {
			scores[index] = score
		}
		first = end
	}
	return scores, median
}

func splitRHat(scores []float64, half int) float64 {
	means := make([]float64, len(scores)/half)
	var within, grand float64
	for i := range means {
		chain := scores[i*half : (i+1)*half]
		var sum float64
		for _, score := range chain {
			sum += score - chain[0]
		}
		mean := chain[0] + sum/float64(half)
		means[i] = mean
		grand += mean
		for _, score := range chain {
			delta := score - mean
			within += delta * delta
		}
	}
	within /= float64(len(means)) * float64(half-1)
	grand /= float64(len(means))
	var between float64
	for _, mean := range means {
		delta := mean - grand
		between += delta * delta
	}
	between /= float64(len(means) - 1)
	return math.Sqrt((float64(half-1)/float64(half)*within + between) / within)
}
