package model

type Options struct {
	Iterations   int
	BurnIn       int
	Thin         int
	Chains       int
	Seed         uint64
	Adapt        bool
	TargetAccept float64
}
