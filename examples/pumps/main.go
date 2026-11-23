package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	t := []float64{94.3, 15.7, 62.9, 126, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5}
	x := []float64{5, 1, 5, 14, 3, 19, 1, 1, 4, 22}
	variables := []*variable.Variable{
		{Name: "alpha", Value: 1, Prior: distributions.Exponential{Lambda: 1}, ProposalScale: .3, Multiplicative: true},
		{Name: "beta", Value: 1, Prior: distributions.Gamma{Alpha: .1, Beta: 1}, ProposalScale: .3, Multiplicative: true},
	}
	names := make([]string, len(x))
	for i := range x {
		name := fmt.Sprintf("theta[%d]", i+1)
		names[i] = name
		variables = append(variables, &variable.Variable{Name: name, Value: (x[i] + .5) / t[i], Prior: distributions.Gamma{Alpha: 1, Beta: 1}, ProposalScale: .3, Multiplicative: true})
	}
	m := &model.Model{Variables: variables, LogLik: func(p map[string]float64) float64 {
		conditional := distributions.Gamma{Alpha: p["alpha"], Beta: p["beta"]}
		var lp float64
		for i, name := range names {
			theta := p[name]
			lp += conditional.LogProb(theta) - (distributions.Gamma{Alpha: 1, Beta: 1}).LogProb(theta)
			lp += (distributions.Poisson{Lambda: theta * t[i]}).LogProb(x[i])
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
	monitorNames := r.Names
	values := make([][][]float64, len(monitorNames))
	for j := range monitorNames {
		values[j] = make([][]float64, len(r.Draws))
		for c, chain := range r.Draws {
			values[j][c] = chain[j]
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
	bayesio.PrintDiagnostics(r)

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
