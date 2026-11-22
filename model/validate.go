package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"

	"github.com/mrtkp9993/gobayes/distributions"
)

func (m *Model) validate(o Options) error {
	var errs []error
	if m == nil {
		return errors.New("model is nil")
	}
	if m.LogLik == nil {
		errs = append(errs, errors.New("LogLik is nil"))
	}
	if len(m.Variables) == 0 {
		errs = append(errs, errors.New("model has no variables"))
	}
	seen := make(map[string]bool)
	for i, v := range m.Variables {
		if v == nil {
			errs = append(errs, fmt.Errorf("variable %d is nil", i))
			continue
		}
		if v.Name == "" {
			errs = append(errs, fmt.Errorf("variable %d has no name", i))
		} else if seen[v.Name] {
			errs = append(errs, fmt.Errorf("duplicate variable name %q", v.Name))
		}
		seen[v.Name] = true
		if v.Prior == nil || (reflect.ValueOf(v.Prior).Kind() == reflect.Ptr && reflect.ValueOf(v.Prior).IsNil()) {
			errs = append(errs, fmt.Errorf("%s: Prior is nil", v.Name))
		}
		_, discrete := v.Prior.(distributions.DiscreteDistribution)
		if discrete && v.Multiplicative {
			errs = append(errs, fmt.Errorf("%s: discrete prior does not support multiplicative proposals", v.Name))
		}
		if !(discrete && v.ProposalScale == 0) && (!(v.ProposalScale > 0) || math.IsInf(v.ProposalScale, 0)) {
			errs = append(errs, fmt.Errorf("%s: ProposalScale must be finite and positive", v.Name))
		}
		if math.IsNaN(v.Value) || math.IsInf(v.Value, 0) {
			errs = append(errs, fmt.Errorf("%s: Value must be finite", v.Name))
		} else if discrete && (math.Trunc(v.Value) != v.Value || math.Abs(v.Value) > 1<<53) {
			errs = append(errs, fmt.Errorf("%s: discrete Value must be an integer between -2^53 and 2^53", v.Name))
		} else if v.Multiplicative && v.Value <= 0 {
			errs = append(errs, fmt.Errorf("%s: multiplicative proposal needs Value > 0", v.Name))
		}
	}
	if o.Iterations <= 0 || o.BurnIn < 0 || o.Thin < 1 || o.Chains < 1 {
		errs = append(errs, errors.New("invalid Iterations/BurnIn/Thin/Chains"))
	} else if o.Iterations > (int(^uint(0)>>1)-o.BurnIn)/o.Thin {
		errs = append(errs, errors.New("iteration count overflows int"))
	}
	if o.Iterations > 0 && len(m.Variables) > 0 && o.Iterations > int(^uint(0)>>1)/len(m.Variables) {
		errs = append(errs, errors.New("draw storage size overflows int"))
	}
	if o.Componentwise && o.Iterations > 0 && o.Thin > 0 && len(m.Variables) > 0 && o.Iterations <= int(^uint(0)>>1)/o.Thin && o.Iterations*o.Thin > int(^uint(0)>>1)/len(m.Variables) {
		errs = append(errs, errors.New("component proposal count overflows int"))
	}
	if math.IsNaN(o.TargetAccept) || math.IsInf(o.TargetAccept, 0) || o.TargetAccept < 0 || o.TargetAccept >= 1 {
		errs = append(errs, errors.New("target acceptance must be zero or between 0 and 1"))
	}
	return errors.Join(errs...)
}
