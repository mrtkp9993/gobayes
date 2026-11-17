package disttest

import (
	"math"
	"testing"
)

const IntegralTol = 1e-10

func deIntegrate(t *testing.T, logTerm func(tt float64) float64, tMax float64) float64 {
	t.Helper()
	sumAt := func(h float64, odd bool) float64 {
		var s float64
		start, step := 0.0, h
		if odd {
			start, step = h, 2*h
		}
		for k := 0; ; k++ {
			tt := start + float64(k)*step
			if tt > tMax {
				break
			}
			for _, sgn := range []float64{1, -1} {
				if tt == 0 && sgn < 0 {
					continue
				}
				v := math.Exp(logTerm(sgn * tt))
				if math.IsNaN(v) || math.IsInf(v, 0) {
					t.Fatalf("integrand not finite at t=%v: %v", sgn*tt, v)
				}
				s += v
			}
		}
		return s
	}
	h := 0.5
	sum := sumAt(h, false)
	prev := sum * h
	for level := 0; level < 10; level++ {
		h /= 2
		sum += sumAt(h, true)
		cur := sum * h
		if math.Abs(cur-prev) < 1e-14 {
			return cur
		}
		prev = cur
	}
	return prev
}

func IntegrateFinite(t *testing.T, d Dist, a, b float64) float64 {
	t.Helper()
	c, r := (a+b)/2, (b-a)/2
	return deIntegrate(t, func(tt float64) float64 {
		u := math.Pi / 2 * math.Sinh(tt)
		x := c + r*math.Tanh(u)
		ch := math.Cosh(u)
		return d.LogProb(x) + math.Log(r*math.Pi/2*math.Cosh(tt)/(ch*ch))
	}, 4)
}

func IntegrateHalfLine(t *testing.T, d Dist, a float64) float64 {
	t.Helper()
	return deIntegrate(t, func(tt float64) float64 {
		u := math.Pi / 2 * math.Sinh(tt)
		x := a + math.Exp(u)
		return d.LogProb(x) + u + math.Log(math.Pi/2*math.Cosh(tt))
	}, 6)
}

func IntegrateRealLine(t *testing.T, d Dist, c, s float64) float64 {
	t.Helper()
	return deIntegrate(t, func(tt float64) float64 {
		u := math.Pi / 2 * math.Sinh(tt)
		x := c + s*math.Sinh(u)
		return d.LogProb(x) + math.Log(s*math.Cosh(u)*math.Pi/2*math.Cosh(tt))
	}, 4.5)
}

func CheckIntegral(t *testing.T, got float64) {
	t.Helper()
	if math.IsNaN(got) || math.Abs(got-1) > IntegralTol {
		t.Errorf("∫ exp(LogProb) dx = %.15g, want 1 (tol %g)", got, IntegralTol)
	}
}
