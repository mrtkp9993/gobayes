package utils

import "math"

// LogPositive computes a logarithm after scaling subnormal positive inputs.
// It uses log(x) = log(x*2^52) - 52*log(2). The same power-of-two
// normalization is used for subnormal inputs in musl's log implementation:
// https://git.musl-libc.org/cgit/musl/tree/src/math/log.c#n71
func LogPositive(x float64) float64 {
	if x > 0 && x < 0x1p-1022 {
		return math.Log(x*0x1p52) - 52*math.Ln2
	}
	return math.Log(x)
}
