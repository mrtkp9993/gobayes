package disttest

import (
	"math"
	"testing"
)

type LogProbCase struct {
	Name string
	X    float64
	Want float64
}

func AssertClose(t *testing.T, got, want float64, label string) {
	t.Helper()
	if math.IsNaN(got) {
		t.Errorf("%s: got NaN, want %v", label, want)
		return
	}
	if math.IsInf(want, 0) || math.IsInf(got, 0) {
		if got != want {
			t.Errorf("%s: got %v, want %v", label, got, want)
		}
		return
	}
	const tol = 1e-12
	if diff := math.Abs(got - want); diff > tol*math.Max(1, math.Abs(want)) {
		t.Errorf("%s: got %.17g, want %.17g (diff %g)", label, got, want, diff)
	}
}

func RunLogProbCases(t *testing.T, d Dist, cases []LogProbCase) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.Name, func(t *testing.T) {
			AssertClose(t, d.LogProb(c.X), c.Want, "LogProb")
		})
	}
}

func AssertPanics(t *testing.T, f func(), label string) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", label)
		}
	}()
	f()
}
