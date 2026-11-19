package model

import "github.com/mrtkp9993/gobayes/variable"

type Model struct {
	Variables []*variable.Variable
	LogLik    func(params map[string]float64) float64
}
