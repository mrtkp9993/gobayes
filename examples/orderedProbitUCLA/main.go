// UCLA socioeconomic status: ses ~ female + read + math.
package main

import (
	"fmt"
	"math"

	"github.com/mrtkp9993/gobayes/distributions"
	bayesio "github.com/mrtkp9993/gobayes/internal/io"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

func main() {
	// https://stats.oarc.ucla.edu/mplus/seminars/mplus-class-notes/analyze/
	// All observations are embedded; Y uses categories 0,1,2.
	// Columns: ses-1, female, read, math.
	data := []struct{ Y, X1, X2, X3 float64 }{
		{0, 1, 34, 41},
		{1, 0, 34, 41},
		{2, 0, 39, 44},
		{0, 0, 37, 42},
		{1, 0, 39, 40},
		{2, 1, 42, 42},
		{1, 0, 31, 46},
		{1, 0, 50, 40},
		{1, 1, 39, 33},
		{1, 0, 34, 46},
		{0, 1, 34, 40},
		{2, 0, 39, 38},
		{0, 0, 47, 44},
		{1, 1, 36, 37},
		{0, 1, 35, 40},
		{0, 0, 44, 39},
		{0, 1, 28, 43},
		{1, 1, 42, 38},
		{1, 0, 34, 45},
		{2, 0, 34, 39},
		{1, 1, 42, 42},
		{1, 0, 37, 45},
		{0, 1, 41, 40},
		{0, 1, 44, 40},
		{0, 1, 47, 43},
		{2, 0, 44, 49},
		{0, 1, 44, 44},
		{1, 0, 44, 46},
		{0, 1, 47, 46},
		{0, 1, 44, 41},
		{1, 1, 47, 42},
		{0, 0, 47, 47},
		{0, 0, 47, 43},
		{0, 1, 47, 49},
		{1, 0, 44, 40},
		{1, 0, 42, 39},
		{1, 0, 50, 49},
		{2, 1, 41, 42},
		{0, 0, 42, 43},
		{1, 0, 47, 41},
		{1, 0, 42, 57},
		{2, 0, 50, 50},
		{0, 1, 45, 44},
		{2, 0, 50, 39},
		{0, 1, 39, 52},
		{0, 1, 42, 41},
		{1, 1, 47, 39},
		{0, 1, 39, 39},
		{1, 1, 47, 52},
		{1, 1, 44, 48},
		{1, 1, 50, 45},
		{1, 0, 55, 40},
		{0, 1, 44, 46},
		{0, 0, 45, 50},
		{1, 1, 44, 43},
		{0, 0, 42, 43},
		{2, 0, 47, 54},
		{2, 1, 36, 42},
		{0, 1, 47, 45},
		{2, 0, 44, 54},
		{1, 1, 42, 47},
		{1, 0, 50, 45},
		{2, 1, 47, 43},
		{1, 1, 43, 40},
		{1, 0, 48, 52},
		{1, 1, 47, 52},
		{0, 1, 42, 45},
		{1, 1, 52, 49},
		{1, 1, 50, 53},
		{1, 1, 39, 54},
		{2, 1, 42, 50},
		{1, 1, 47, 48},
		{2, 0, 52, 51},
		{0, 0, 57, 41},
		{1, 1, 42, 51},
		{0, 1, 47, 46},
		{1, 1, 46, 55},
		{1, 1, 50, 46},
		{1, 0, 44, 61},
		{1, 0, 50, 45},
		{0, 0, 36, 54},
		{1, 1, 39, 54},
		{2, 0, 47, 51},
		{0, 0, 52, 49},
		{2, 1, 50, 56},
		{0, 1, 50, 53},
		{1, 1, 47, 49},
		{1, 0, 55, 57},
		{1, 0, 50, 42},
		{1, 0, 55, 46},
		{2, 1, 50, 45},
		{2, 1, 43, 55},
		{2, 0, 63, 47},
		{1, 1, 57, 50},
		{1, 1, 50, 41},
		{1, 1, 55, 52},
		{1, 0, 47, 57},
		{1, 1, 50, 53},
		{1, 0, 60, 51},
		{1, 1, 57, 57},
		{1, 0, 44, 51},
		{1, 1, 57, 40},
		{1, 1, 52, 50},
		{1, 0, 42, 57},
		{1, 0, 63, 35},
		{0, 1, 61, 49},
		{0, 1, 60, 50},
		{1, 1, 52, 55},
		{1, 1, 52, 48},
		{1, 0, 57, 52},
		{0, 1, 47, 53},
		{1, 0, 57, 59},
		{1, 1, 55, 66},
		{1, 0, 52, 57},
		{1, 1, 47, 54},
		{1, 0, 47, 57},
		{2, 0, 55, 54},
		{2, 1, 63, 54},
		{1, 1, 57, 54},
		{1, 1, 60, 49},
		{0, 1, 57, 51},
		{1, 1, 52, 58},
		{1, 1, 47, 60},
		{2, 1, 47, 51},
		{1, 0, 47, 63},
		{2, 0, 52, 50},
		{1, 1, 50, 53},
		{1, 1, 52, 53},
		{1, 1, 55, 55},
		{2, 1, 55, 56},
		{1, 0, 63, 49},
		{2, 0, 55, 61},
		{0, 0, 63, 49},
		{2, 1, 65, 57},
		{1, 0, 52, 66},
		{2, 1, 47, 56},
		{1, 0, 60, 60},
		{2, 0, 54, 57},
		{0, 0, 63, 59},
		{2, 0, 60, 58},
		{2, 0, 60, 57},
		{0, 1, 52, 64},
		{1, 0, 57, 60},
		{1, 0, 63, 54},
		{1, 0, 53, 61},
		{1, 1, 55, 58},
		{1, 1, 57, 56},
		{0, 1, 52, 60},
		{1, 0, 63, 55},
		{2, 0, 63, 57},
		{1, 0, 55, 62},
		{2, 1, 63, 56},
		{1, 0, 57, 51},
		{1, 1, 68, 56},
		{0, 1, 50, 61},
		{1, 1, 57, 63},
		{2, 1, 65, 61},
		{2, 1, 60, 67},
		{0, 0, 63, 48},
		{2, 0, 47, 61},
		{2, 1, 52, 57},
		{2, 0, 65, 48},
		{0, 1, 63, 65},
		{2, 1, 60, 62},
		{1, 1, 68, 61},
		{1, 1, 68, 53},
		{2, 0, 60, 58},
		{2, 0, 55, 64},
		{2, 1, 63, 65},
		{2, 0, 68, 56},
		{0, 0, 55, 63},
		{2, 1, 73, 57},
		{0, 1, 57, 72},
		{2, 1, 50, 66},
		{2, 0, 68, 62},
		{0, 1, 68, 58},
		{1, 1, 65, 63},
		{0, 1, 65, 64},
		{0, 1, 57, 72},
		{2, 0, 76, 64},
		{2, 1, 63, 69},
		{1, 0, 65, 70},
		{2, 0, 65, 66},
		{1, 0, 68, 58},
		{2, 1, 73, 62},
		{2, 1, 66, 67},
		{2, 1, 68, 64},
		{2, 0, 65, 63},
		{2, 0, 65, 68},
		{1, 0, 68, 75},
		{2, 1, 71, 69},
		{2, 1, 68, 65},
		{1, 0, 68, 71},
		{2, 0, 73, 71},
		{2, 1, 76, 60},
		{2, 1, 63, 71},
		{1, 0, 63, 75},
		{1, 0, 73, 71},
		{1, 1, 71, 72},
		{1, 0, 73, 73},
	}
	type groupedRow struct {
		Y, X1, X2, X3 float64
		Count         int
	}
	grouped := []groupedRow{}
	indices := make(map[[4]float64]int)
	for _, row := range data {
		row.X2 = (row.X2 - 50) / 10
		row.X3 = (row.X3 - 50) / 10
		key := [4]float64{row.Y, row.X1, row.X2, row.X3}
		if i, ok := indices[key]; ok {
			grouped[i].Count++
		} else {
			indices[key] = len(grouped)
			grouped = append(grouped, groupedRow{row.Y, row.X1, row.X2, row.X3, 1})
		}
	}
	m := &model.Model{
		Variables: []*variable.Variable{
			{Name: "b1", Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 2}, ProposalScale: 0.2},
			{Name: "b2", Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 2}, ProposalScale: 0.2},
			{Name: "b3", Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 2}, ProposalScale: 0.2},
			{Name: "c1", Value: -0.7, Prior: distributions.Normal{Mu: 0, Sigma: 2}, ProposalScale: 0.2},
			{Name: "gap", Value: 1.3, Prior: distributions.Gamma{Alpha: 2, Beta: 1}, ProposalScale: 0.2, Multiplicative: true},
		},
		LogLik: func(params map[string]float64) float64 {
			cuts := []float64{params["c1"], params["c1"] + params["gap"]}
			b1, b2, b3 := params["b1"], params["b2"], params["b3"]
			var lp float64
			for _, row := range grouped {
				eta := b1*row.X1 + b2*row.X2 + b3*row.X3
				lp += float64(row.Count) * (distributions.OrderedProbit{Eta: eta, Cutpoints: cuts}).LogProb(row.Y)
			}
			return lp
		},
	}
	result, err := m.Run(model.Options{Iterations: 20000, BurnIn: 10000, Chains: 4, Seed: 42,
		Adapt: true, Componentwise: true, TargetAccept: 0.44})
	if err != nil {
		panic(err)
	}
	// Report slopes and thresholds on the original covariate scale.
	monitored := *result
	monitored.Names = []string{"beta_female", "beta_read", "beta_math", "cut1", "cut2"}
	monitored.Draws = make([][][]float64, len(result.Draws))
	for c, chain := range result.Draws {
		monitored.Draws[c] = make([][]float64, 5)
		for k := range monitored.Names {
			monitored.Draws[c][k] = make([]float64, len(chain[0]))
		}
		for i := range chain[0] {
			b1, b2, b3, c1, gap := chain[0][i], chain[1][i], chain[2][i], chain[3][i], chain[4][i]
			c1 += 5 * (b2 + b3)
			b2 /= 10
			b3 /= 10
			values := [5]float64{b1, b2, b3, c1, c1 + gap}
			for k, x := range values {
				monitored.Draws[c][k][i] = x
			}
		}
	}
	fmt.Printf("N=%d; 4 chains, seed=42, warmup=10000, saved=20000 per chain\n", len(data))
	fmt.Println("Parameter       Mean       SD")
	for k, name := range monitored.Names {
		var sum, squared float64
		n := 0
		for _, chain := range monitored.Draws {
			for _, x := range chain[k] {
				sum += x
				n++
			}
		}
		mean := sum / float64(n)
		for _, chain := range monitored.Draws {
			for _, x := range chain[k] {
				squared += (x - mean) * (x - mean)
			}
		}
		fmt.Printf("%-14s %.6f   %.6f\n", name, mean, math.Sqrt(squared/float64(n-1)))
	}
	bayesio.PrintDiagnostics(&monitored)
}
