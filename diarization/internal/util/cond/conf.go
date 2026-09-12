package cond

func Ternary[T any](cond bool, val1, val2 T) T {
	if cond {
		return val1
	}
	return val2
}
