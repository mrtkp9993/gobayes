package diagnostics

type ParameterDiagnostics struct {
	Autocorrelation []float64
	ESS             float64
	MCSEMean        float64
	RHat            *float64 // nil if there is only one chain
}
