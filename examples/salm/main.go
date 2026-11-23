// Salm is the OpenBUGS Poisson log-normal dose-response model.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	x := []float64{0, 10, 33, 100, 333, 1000}
	y := [][3]float64{{15, 21, 29}, {16, 18, 21}, {16, 26, 33}, {27, 41, 60}, {33, 38, 41}, {20, 27, 42}}
	logX := make([]float64, len(x))
	var meanLogX, meanX float64
	for i := range x {
		logX[i] = math.Log(x[i] + 10)
		meanLogX += logX[i] / float64(len(x))
		meanX += x[i] / float64(len(x))
	}
	variables := []*variable.Variable{
		{Name: "intercept", Value: 3.3, Prior: distributions.Normal{Mu: 0, Sigma: 1000}, ProposalScale: .1},
		{Name: "beta", Value: .3, Prior: distributions.Normal{Mu: 0, Sigma: 1000}, ProposalScale: .1},
		{Name: "g", Value: -1, Prior: distributions.Normal{Mu: 0, Sigma: 1e6}, ProposalScale: .2},
		{Name: "tau", Value: 10, Prior: distributions.Gamma{Alpha: .001, Beta: .001}, ProposalScale: .3, Multiplicative: true},
	}
	names := make([][3]string, len(x))
	for i := range x {
		for j := 0; j < 3; j++ {
			name := fmt.Sprintf("z[%d,%d]", i+1, j+1)
			names[i][j] = name
			variables = append(variables, &variable.Variable{Name: name, Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 1}, ProposalScale: .5})
		}
	}
	m := &model.Model{Variables: variables, LogLik: func(p map[string]float64) float64 {
		prior := distributions.Normal{Mu: 0, Sigma: 1000}
		alpha := p["intercept"] - p["beta"]*meanLogX - p["g"]*meanX/1000
		lp := prior.LogProb(alpha) - prior.LogProb(p["intercept"])
		sigma := 1 / math.Sqrt(p["tau"])
		for i := range x {
			for j := 0; j < 3; j++ {
				eta := p["intercept"] + p["beta"]*(logX[i]-meanLogX) + p["g"]*(x[i]-meanX)/1000 + sigma*p[names[i][j]]
				lp += (distributions.Poisson{Lambda: math.Exp(eta)}).LogProb(y[i][j])
			}
		}
		return lp
	}}
	iterations := flag.Int("iterations", 10000, "saved draws per chain")
	warmup := flag.Int("warmup", 5000, "warmup sweeps per chain")
	chains := flag.Int("chains", 4, "number of chains")
	seed := flag.Uint64("seed", 20261123, "random seed")
	output := flag.String("csv", "", "optional monitored draws CSV path")
	flag.Parse()
	r, err := m.Run(model.Options{Iterations: *iterations, BurnIn: *warmup, Chains: *chains,
		Seed: *seed, Adapt: true, TargetAccept: 0.44, Componentwise: true})
	if err != nil {
		panic(err)
	}
	monitorNames := []string{"alpha", "beta", "gamma", "sigma"}
	values := make([][][]float64, len(monitorNames))
	for j := range monitorNames {
		values[j] = make([][]float64, len(r.Draws))
		for c, chain := range r.Draws {
			values[j][c] = make([]float64, len(chain[0]))
		}
	}
	for c, chain := range r.Draws {
		for i := range chain[0] {
			values[0][c][i] = chain[0][i] - chain[1][i]*meanLogX - chain[2][i]*meanX/1000
			values[1][c][i] = chain[1][i]
			values[2][c][i] = chain[2][i] / 1000
			values[3][c][i] = 1 / math.Sqrt(chain[3][i])
		}
	}

	for j, name := range monitorNames {
		var sum float64
		count := 0
		for _, chain := range values[j] {
			for _, value := range chain {
				sum += value
				count++
			}
		}
		fmt.Printf("Posterior mean of %s ≈ %.6f\n", name, sum/float64(count))
	}
	// Report diagnostics on the original model scale.
	summary := *r
	summary.Names = monitorNames
	summary.Draws = make([][][]float64, len(r.Draws))
	for c := range summary.Draws {
		summary.Draws[c] = make([][]float64, len(monitorNames))
		for j := range monitorNames {
			summary.Draws[c][j] = values[j][c]
		}
	}
	bayesio.PrintDiagnostics(&summary)

	if *output == "" {
		return
	}
	f, err := os.Create(*output)
	if err != nil {
		panic(err)
	}
	w := csv.NewWriter(f)
	header := []string{"chain", "draw"}
	for _, name := range monitorNames {
		header = append(header, name)
	}
	if err := w.Write(header); err != nil {
		panic(err)
	}
	for c := range r.Draws {
		for i := range r.Draws[c][0] {
			row := []string{strconv.Itoa(c + 1), strconv.Itoa(i + 1)}
			for j := range monitorNames {
				row = append(row, strconv.FormatFloat(values[j][c][i], 'g', 17, 64))
			}
			if err := w.Write(row); err != nil {
				panic(err)
			}
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		panic(err)
	}
	if err := f.Close(); err != nil {
		panic(err)
	}
}
