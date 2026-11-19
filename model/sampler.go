package model

import (
	"github.com/mrtkp9993/gobayes/random"
	"math"
)

func (m *Model) initialState() Proposal {
	s := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		s.Params[v.Name] = v.Value
		s.LogPrior += v.Prior.LogProb(v.Value)
	}
	if !math.IsNaN(s.LogPrior) && !math.IsInf(s.LogPrior, 0) {
		s.LogLik = m.LogLik(s.Params)
	}
	return s
}

func (m *Model) proposeWithScales(curr Proposal, rng *random.Generator, scales []float64, p *Proposal) {
	p.LogPrior, p.LogLik, p.LogQRatio = 0, 0, 0
	for i, v := range m.Variables {
		scale := v.ProposalScale
		if scales != nil {
			scale = scales[i]
		}
		var newVal float64
		if v.Multiplicative {
			factor := math.Exp(scale * rng.NormFloat64())
			newVal = curr.Params[v.Name] * factor
			p.LogQRatio += math.Log(factor)
		} else {
			newVal = curr.Params[v.Name] + scale*rng.NormFloat64()
		}
		p.Params[v.Name] = newVal
		p.LogPrior += v.Prior.LogProb(newVal)
	}
}

type acceptanceDecision uint8

const (
	accepted acceptanceDecision = iota
	rejectedProbability
	rejectedPrior
	rejectedNonFinite
)

func (m *Model) acceptDecision(p *Proposal, curr Proposal, rng *random.Generator) acceptanceDecision {
	if math.IsInf(p.LogPrior, -1) {
		return rejectedPrior
	}
	if math.IsNaN(p.LogPrior) || math.IsInf(p.LogPrior, 1) {
		return rejectedNonFinite
	}
	p.LogLik = m.LogLik(p.Params)
	if math.IsNaN(p.LogLik) || math.IsInf(p.LogLik, 1) || math.IsNaN(p.LogQRatio) || math.IsInf(p.LogQRatio, 0) {
		return rejectedNonFinite
	}
	if math.IsInf(p.LogLik, -1) {
		return rejectedProbability
	}
	logA := p.LogPosterior() - curr.LogPosterior() + p.LogQRatio
	if math.IsNaN(logA) {
		return rejectedNonFinite
	}
	if logA >= 0 || math.Log(rng.Float64()) < logA {
		return accepted
	}
	return rejectedProbability
}
