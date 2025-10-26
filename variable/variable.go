package variable

import (
	"github.com/mrtkp9993/gobayes/distributions"
)

type Variable struct {
	Name           string
	Value          float64
	Prior          distributions.Distribution
	ProposalScale  float64
	Multiplicative bool
}
