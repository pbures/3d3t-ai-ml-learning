package agents

import (
	"testing"
	"time"
)

func TestAcquireDeepSearchSlotBoundsConcurrency(t *testing.T) {
	var releases []func()
	for i := 0; i < MaxConcurrentDeepSearches; i++ {
		rel, ok := acquireDeepSearchSlot(time.Now().Add(100 * time.Millisecond))
		if !ok {
			t.Fatalf("slot %d: expected immediate acquisition", i)
		}
		releases = append(releases, rel)
	}

	// All slots taken: acquisition must time out.
	start := time.Now()
	if _, ok := acquireDeepSearchSlot(time.Now().Add(30 * time.Millisecond)); ok {
		t.Fatal("expected acquisition to fail while all slots are held")
	}
	if time.Since(start) < 25*time.Millisecond {
		t.Fatal("expected acquisition to wait until the deadline")
	}

	// A slot freed while waiting is picked up before the deadline.
	go func() {
		time.Sleep(20 * time.Millisecond)
		releases[0]()
	}()
	rel, ok := acquireDeepSearchSlot(time.Now().Add(time.Second))
	if !ok {
		t.Fatal("expected acquisition after a slot was released")
	}
	rel()
	for _, r := range releases[1:] {
		r()
	}
}
