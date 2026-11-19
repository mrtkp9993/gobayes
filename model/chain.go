package model

import (
	"github.com/mrtkp9993/gobayes/random"
	"math"
)

func (m *Model) runChain(o Options, rng *random.Generator, r *Result, c int, curr Proposal) error {
	r.Draws[c] = make([][]float64, len(m.Variables))
	backing := make([]float64, len(m.Variables)*o.Iterations)
	for i := range r.Draws[c] {
		r.Draws[c][i] = backing[i*o.Iterations : (i+1)*o.Iterations : (i+1)*o.Iterations]
	}
	proposal := Proposal{Params: make(map[string]float64, len(m.Variables))}
	r.Scales[c] = make([]float64, len(m.Variables))
	for i, v := range m.Variables {
		r.Scales[c][i] = v.ProposalScale
	}
	for step := 0; step < o.BurnIn+o.Iterations*o.Thin; step++ {
		m.proposeWithScales(curr, rng, r.Scales[c], &proposal)
		decision := m.acceptDecision(&proposal, curr, rng)
		if step >= o.BurnIn {
			d := &r.Diagnostics[c]
			d.Proposed++
			switch decision {
			case accepted:
				d.Accepted++
			case rejectedProbability:
				d.RejectedProbability++
			case rejectedPrior:
				d.RejectedPrior++
			case rejectedNonFinite:
				d.RejectedNonFinite++
			}
		}
		acceptedProposal := decision == accepted
		if acceptedProposal {
			curr, proposal = proposal, curr
		}
		if o.Adapt && step < o.BurnIn {
			indicator := 0.0
			if acceptedProposal {
				indicator = 1
			}
			gain := 1 / math.Sqrt(float64(step+1))
			for i, scale := range r.Scales[c] {
				logScale := math.Log(scale) + gain*(indicator-o.TargetAccept)
				r.Scales[c][i] = math.Exp(math.Max(-20, math.Min(20, logScale)))
			}
		}
		if step >= o.BurnIn && (step-o.BurnIn+1)%o.Thin == 0 {
			drawIndex := (step-o.BurnIn+1)/o.Thin - 1
			for i, name := range r.Names {
				r.Draws[c][i][drawIndex] = curr.Params[name]
			}
		}
	}
	return nil
}
