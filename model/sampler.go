package model

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/random"
)

func (m *Model) initialState(rng *random.Generator) Proposal {
	s := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		value := v.Value
		if rng != nil {
			if _, discrete := v.Prior.(distributions.DiscreteDistribution); discrete {
				value = proposeDiscrete(value, v.ProposalScale, rng)
			} else {
				step := v.ProposalScale * (2*rng.Float64() - 1)
				if v.Multiplicative {
					value *= math.Exp(step)
				} else {
					value += step
				}
			}
		}
		if math.IsNaN(value) || math.IsInf(value, 0) || (v.Multiplicative && value <= 0) {
			s.LogPrior = math.Inf(-1)
			return s
		}
		s.Params[v.Name] = value
		s.LogPrior += v.Prior.LogProb(value)
	}
	if !math.IsNaN(s.LogPrior) && !math.IsInf(s.LogPrior, 0) {
		s.LogLik = m.LogLik(s.Params)
	}
	return s
}

func (m *Model) randomizedInitialState(rng *random.Generator, previous []Proposal) (Proposal, error) {
	const maxAttempts = 1000
	var fallback *Proposal
	for range maxAttempts {
		candidate := m.initialState(rng)
		if lp := candidate.LogPosterior(); math.IsNaN(lp) || math.IsInf(lp, 0) {
			continue
		}
		if slices.ContainsFunc(previous, func(start Proposal) bool {
			return maps.Equal(start.Params, candidate.Params)
		}) {
			if m.hasDiscrete() && fallback == nil {
				fallback = &candidate
			}
			continue
		}
		return candidate, nil
	}
	if fallback != nil {
		return *fallback, nil
	}
	return Proposal{}, fmt.Errorf("could not find a distinct initial state with finite log-posterior after %d attempts", maxAttempts)
}

func (m *Model) proposeWithScales(curr Proposal, rng *random.Generator, scales []float64, p *Proposal) {
	p.LogPrior, p.LogLik, p.LogQRatio = 0, 0, 0
	for i, v := range m.Variables {
		scale := v.ProposalScale
		if scales != nil {
			scale = scales[i]
		}
		var newVal float64
		if _, discrete := v.Prior.(distributions.DiscreteDistribution); discrete {
			newVal = proposeDiscrete(curr.Params[v.Name], scale, rng)
		} else {
			var logQRatio float64
			newVal, logQRatio = proposeContinuous(curr.Params[v.Name], scale, v.Multiplicative, rng)
			p.LogQRatio += logQRatio
		}
		p.Params[v.Name] = newVal
		if math.IsNaN(newVal) || math.IsInf(newVal, 0) {
			p.LogPrior = math.Inf(-1)
			return
		}
		p.LogPrior += v.Prior.LogProb(newVal)
	}
}

func (m *Model) proposeComponent(curr Proposal, rng *random.Generator, scale float64, i int, p *Proposal) {
	for name, value := range curr.Params {
		p.Params[name] = value
	}
	v := m.Variables[i]
	old := curr.Params[v.Name]
	var value float64
	p.LogQRatio = 0
	if _, discrete := v.Prior.(distributions.DiscreteDistribution); discrete {
		value = proposeDiscrete(old, scale, rng)
	} else {
		value, p.LogQRatio = proposeContinuous(old, scale, v.Multiplicative, rng)
	}
	p.Params[v.Name] = value
	if math.IsNaN(value) || math.IsInf(value, 0) {
		p.LogPrior = math.Inf(-1)
		p.LogLik = 0
		return
	}
	p.LogPrior = 0
	for _, variable := range m.Variables {
		p.LogPrior += variable.Prior.LogProb(p.Params[variable.Name])
	}
	p.LogLik = 0
}

func (m *Model) hasDiscrete() bool {
	for _, v := range m.Variables {
		if _, ok := v.Prior.(distributions.DiscreteDistribution); ok {
			return true
		}
	}
	return false
}

func proposeContinuous(value, scale float64, multiplicative bool, rng *random.Generator) (candidate, logQRatio float64) {
	step := scale * rng.NormFloat64()
	if multiplicative {
		return value * math.Exp(step), step
	}
	return value + step, 0
}

func proposeDiscrete(value, scale float64, rng *random.Generator) float64 {
	if scale == 0 {
		scale = 1
	}
	step := math.Round(scale * rng.NormFloat64())
	const maxInteger = 1 << 53
	if step < -maxInteger-value || step > maxInteger-value {
		return math.Inf(1)
	}
	return value + step
}

func (m *Model) accept(p *Proposal, curr Proposal, deltaLogPrior float64, rng *random.Generator) bool {
	if math.IsInf(p.LogPrior, -1) {
		return false
	}
	if math.IsNaN(p.LogPrior) || math.IsInf(p.LogPrior, 1) {
		return false
	}
	p.LogLik = m.LogLik(p.Params)
	if math.IsNaN(p.LogLik) || math.IsInf(p.LogLik, 1) || math.IsNaN(p.LogQRatio) || math.IsInf(p.LogQRatio, 0) {
		return false
	}
	if math.IsInf(p.LogLik, -1) {
		return false
	}
	logA := (p.LogLik - curr.LogLik) + deltaLogPrior + p.LogQRatio
	if math.IsNaN(logA) {
		return false
	}
	if logA >= 0 || math.Log(rng.Float64()) < logA {
		return true
	}
	return false
}
