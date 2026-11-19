package diagnostics

type ChainDiagnostics struct {
	Proposed          int
	Accepted          int
	RejectedPrior     int
	RejectedNonFinite int
}
