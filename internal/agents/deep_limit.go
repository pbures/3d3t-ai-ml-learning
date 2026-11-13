package agents

import "time"

// MaxConcurrentDeepSearches bounds how many DeepSearchAgent searches may hold their transposition
// tables at the same time. Each search needs a 16 MiB PVS table plus (late in the game) a 64 MiB
// WDL solver table, so without a bound N concurrent games cost ~N*80 MiB of live heap (plus GC
// headroom) and can OOM-kill a memory-limited container.
const MaxConcurrentDeepSearches = 2

var deepSearchSlots = make(chan struct{}, MaxConcurrentDeepSearches)

// acquireDeepSearchSlot waits for a free search slot until `deadline`. It returns a release func
// and true on success, or nil and false if no slot became free in time.
func acquireDeepSearchSlot(deadline time.Time) (func(), bool) {
	release := func() { <-deepSearchSlots }
	select {
	case deepSearchSlots <- struct{}{}:
		return release, true
	default:
	}
	wait := time.Until(deadline)
	if wait <= 0 {
		return nil, false
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case deepSearchSlots <- struct{}{}:
		return release, true
	case <-timer.C:
		return nil, false
	}
}
