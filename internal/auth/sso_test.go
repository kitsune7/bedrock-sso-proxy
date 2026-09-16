package auth

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/aws/smithy-go"
)

// stubAPIError is a smithy.APIError with a chosen error code.
type stubAPIError struct{ code string }

func (e *stubAPIError) Error() string                 { return "api error " + e.code }
func (e *stubAPIError) ErrorCode() string             { return e.code }
func (e *stubAPIError) ErrorMessage() string          { return e.code }
func (e *stubAPIError) ErrorFault() smithy.ErrorFault { return smithy.FaultServer }

// testAWSEnv points the SDK at a throwaway profile with static keys so
// renew's config reload succeeds without touching the real ~/.aws or the network.
func testAWSEnv(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config")
	credPath := filepath.Join(dir, "credentials")
	if err := os.WriteFile(cfgPath, []byte("[profile test]\nregion = us-east-1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cred := "[test]\naws_access_key_id = AKIAEXAMPLE\naws_secret_access_key = examplesecret\n"
	if err := os.WriteFile(credPath, []byte(cred), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("AWS_CONFIG_FILE", cfgPath)
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", credPath)
}

// A batch of requests failing on the same expired token must produce exactly one
// `aws sso login`. One browser tab per in-flight request was the original bug:
// every device code but the approved one died with InvalidGrantException.
func TestRenewIsSingleFlightAcrossConcurrentRequests(t *testing.T) {
	testAWSEnv(t)

	const callers = 8
	var logins, attempts int32
	release := make(chan struct{})
	started := make(chan struct{}, callers)

	am := &AuthManager{
		profile:   "test",
		region:    "us-east-1",
		loginSlot: make(chan struct{}, 1),
	}
	am.login = func() error {
		atomic.AddInt32(&logins, 1)
		<-release // hold the login open so every other caller queues behind it
		return nil
	}

	errs := make([]error, callers)
	var wg sync.WaitGroup
	for i := range callers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = am.withRetry(context.Background(), func() error {
				// The first attempt from each caller fails; retries succeed.
				if atomic.AddInt32(&attempts, 1) <= callers {
					started <- struct{}{}
					return &stubAPIError{code: "ExpiredTokenException"}
				}
				return nil
			})
		}(i)
	}
	for range callers {
		<-started // every caller has failed and is now inside renew
	}
	close(release)
	wg.Wait()

	if got := atomic.LoadInt32(&logins); got != 1 {
		t.Errorf("want exactly 1 SSO login for %d concurrent failures, got %d", callers, got)
	}
	for i, err := range errs {
		if err != nil {
			t.Errorf("caller %d failed after renewal: %v", i, err)
		}
	}
	if gen := am.generation(); gen != 1 {
		t.Errorf("want generation 1 after one renewal, got %d", gen)
	}
}

// A caller that gives up must not sit behind someone else's browser flow.
func TestRenewHonorsCallerCancellation(t *testing.T) {
	am := &AuthManager{profile: "test", region: "us-east-1", loginSlot: make(chan struct{}, 1)}
	am.login = func() error { t.Fatal("login must not run for a cancelled caller"); return nil }
	am.loginSlot <- struct{}{} // simulate a login already in flight

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := am.renew(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Errorf("want context.Canceled, got %v", err)
	}
}

func TestIsSSOErrorClassification(t *testing.T) {
	needsLogin := []error{
		&stubAPIError{code: "ExpiredTokenException"},
		&stubAPIError{code: "UnrecognizedClientException"},
		errors.New("failed to refresh cached credentials, refresh cached SSO token failed"),
	}
	for _, err := range needsLogin {
		if !isSSOError(err) {
			t.Errorf("want SSO error: %v", err)
		}
	}

	ignored := []error{
		// Authorization, not authentication — an ungranted Bedrock model or a
		// throttle used to open a browser.
		&stubAPIError{code: "AccessDeniedException"},
		&stubAPIError{code: "ThrottlingException"},
		&stubAPIError{code: "ValidationException"},
		// Our own renewal failure must not classify as "needs a login".
		errors.New("SSO re-authentication failed: exit status 254"),
	}
	for _, err := range ignored {
		if isSSOError(err) {
			t.Errorf("want non-SSO error: %v", err)
		}
	}
}
