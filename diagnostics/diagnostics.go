package diagnostics

type Diagnostics struct {
	Chains []ChainDiagnostics
	Params map[string]ParameterDiagnostics
}
