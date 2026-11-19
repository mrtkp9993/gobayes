package diagnostics

type ChainDiagnostics struct {
	Proposed            int
	Accepted            int
	RejectedProbability int
	RejectedPrior       int
	RejectedNonFinite   int
}
