package model

import (
	"errors"
	"math"
	"math/rand/v2"

	"github.com/mrtkp9993/gobayes/variable"
)

type Model struct {
	Variables []*variable.Variable
	LogLik    func(params map[string]float64) float64
}

func (m *Model) logPosterior() float64 {
	params := make(map[string]float64)
	for _, v := range m.Variables {
		params[v.Name] = v.Value
	}
	ll := m.LogLik(params)
	lp := 0.0
	for _, v := range m.Variables {
		lp += v.Prior.LogProb(v.Value)
	}
	return ll + lp
}

func (m *Model) propose() (map[string]float64, float64) {
	proposal := make(map[string]float64)
	logPrior := 0.0
	proposalLogRatio := 0.0
	for _, v := range m.Variables {
		var newVal float64
		if v.Multiplicative {
			factor := math.Exp(v.ProposalScale * rand.NormFloat64())
			newVal = v.Value * factor
			proposalLogRatio += math.Log(factor)
		} else {
			newVal = v.Value + v.ProposalScale*rand.NormFloat64()
		}
		proposal[v.Name] = newVal
		logPrior += v.Prior.LogProb(newVal)
	}
	proposal["__proposalLogRatio"] = proposalLogRatio
	return proposal, logPrior
}

func (m *Model) accept(proposal map[string]float64, lp float64) bool {
	proposalLogRatio := 0.0
	if val, ok := proposal["__proposalLogRatio"]; ok {
		proposalLogRatio = val
		delete(proposal, "__proposalLogRatio")
	}
	llProp := m.LogLik(proposal)
	llCurr := m.LogLik(currentParams(m.Variables))
	lpCurr := 0.0
	for _, v := range m.Variables {
		lpCurr += v.Prior.LogProb(v.Value)
	}
	logA := llProp + lp - (llCurr + lpCurr) + proposalLogRatio
	if logA >= 0 {
		return true
	}
	return rand.Float64() < math.Exp(logA)
}

func currentParams(vars []*variable.Variable) map[string]float64 {
	params := make(map[string]float64)
	for _, v := range vars {
		params[v.Name] = v.Value
	}
	return params
}

func (m *Model) RunMCMC(iterations, burnIn int) ([]map[string]float64, error) {
	if iterations <= 0 {
		return nil, errors.New("iterations must be positive")
	}
	if burnIn < 0 {
		return nil, errors.New("burn-in period cannot be negative")
	}

	samples := make([]map[string]float64, iterations)
	for i := 0; i < iterations+burnIn; i++ {
		proposal, lpProp := m.propose()
		if m.accept(proposal, lpProp) {
			for _, v := range m.Variables {
				v.Value = proposal[v.Name]
			}
		}

		if i >= burnIn {
			samples[i-burnIn] = make(map[string]float64)
			for _, v := range m.Variables {
				samples[i-burnIn][v.Name] = v.Value
			}
		}
	}
	return samples, nil
}
