package exampleutil

import (
	"fmt"
	"os"
	"text/tabwriter"

	"github.com/mrtkp9993/gobayes/model"
)

func PrintDiagnostics(result *model.Result) {
	rates := result.AcceptRate()
	fmt.Println("\nChain diagnostics (after burn-in)")
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "Chain\tAcceptance\tProposed\tAccepted\tProbability reject\tPrior reject\tNon-finite reject")
	for i, d := range result.Diagnostics {
		fmt.Fprintf(w, "%d\t%.3f\t%d\t%d\t%d\t%d\t%d\n",
			i+1, rates[i], d.Proposed, d.Accepted, d.RejectedProbability, d.RejectedPrior, d.RejectedNonFinite)
	}
	w.Flush()
	fmt.Println("\nParameter diagnostics")
	w = tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "Parameter\tESS\tMCSE mean\tR-hat\tACF 0\tACF 1\tACF 2\tACF 3\tACF 4\tACF 5")
	for _, name := range result.Names {
		d, err := result.ParameterDiagnostics(name)
		if err != nil {
			fmt.Fprintf(w, "%s\tunavailable: %v\n", name, err)
			continue
		}
		rhat := "—"
		if d.RHat != nil {
			rhat = fmt.Sprintf("%.4f", *d.RHat)
		}
		fmt.Fprintf(w, "%s\t%.1f\t%.6g\t%s", name, d.ESS, d.MCSEMean, rhat)
		for lag := 0; lag < 6; lag++ {
			if lag < len(d.Autocorrelation) {
				fmt.Fprintf(w, "\t%.3f", d.Autocorrelation[lag])
			} else {
				fmt.Fprint(w, "\t—")
			}
		}
		fmt.Fprintln(w)
	}
	w.Flush()
	if len(result.Draws) == 1 {
		fmt.Println("R-hat requires multiple chains.")
	}
}
