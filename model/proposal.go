package model

type Proposal struct {
	Params    map[string]float64
	LogPrior  float64
	LogLik    float64
	LogQRatio float64
}

func (p Proposal) LogPosterior() float64 {
	return p.LogPrior + p.LogLik
}
