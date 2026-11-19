package auth

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/smithy-go"
)

// roleAccessDenied is the error the proxy saw every time after an `aws sso
// login` that had just succeeded: the SSO portal refusing GetRoleCredentials.
func roleAccessDenied() error {
	return fmt.Errorf("operation error Bedrock Runtime: ConverseStream, get identity: get credentials: "+
		"failed to refresh cached credentials, operation error SSO: GetRoleCredentials, "+
		"https response error StatusCode: 403, RequestID: x, api error ForbiddenException: No access: %w",
		&stubAPIError{code: "ForbiddenException"})
}

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
		&stubAPIError{code: "UnauthorizedException"},
		errors.New("failed to refresh cached credentials, refresh cached SSO token failed"),
		// A 403 from the SSO portal can follow a stale token, so it earns one
		// login; the cooldown tests below cover what stops it repeating.
		roleAccessDenied(),
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
		&stubAPIError{code: "TooManyRequestsException"},
		&stubAPIError{code: "ValidationException"},
		// The SDK wraps throttles in the same "failed to refresh cached
		// credentials" prefix that signals expiry; the typed code must win.
		fmt.Errorf("failed to refresh cached credentials: %w", &stubAPIError{code: "ThrottlingException"}),
		// Our own renewal failure must not classify as "needs a login".
		errors.New("SSO re-authentication failed: exit status 254"),
	}
	for _, err := range ignored {
		if isSSOError(err) {
			t.Errorf("want non-SSO error: %v", err)
		}
	}
}

func TestIsRoleAccessDenied(t *testing.T) {
	if !isRoleAccessDenied(roleAccessDenied()) {
		t.Error("want the GetRoleCredentials 403 recognised")
	}
	for _, err := range []error{
		nil,
		errors.New("ForbiddenException on some other call"),
		&stubAPIError{code: "ExpiredTokenException"},
		errors.New("operation error SSO: GetRoleCredentials, UnauthorizedException"),
	} {
		if isRoleAccessDenied(err) {
			t.Errorf("want not a role-access denial: %v", err)
		}
	}
}

// When a login fails (nobody approved the device code), every request that was
// queued behind it used to run its own login in turn — one 10-minute browser
// wait each. They must all fail on the first failure instead.
func TestFailedLoginDoesNotRepeatForQueuedCallers(t *testing.T) {
	const callers = 6
	var logins int32
	release := make(chan struct{})
	am := &AuthManager{
		profile:             "test",
		region:              "us-east-1",
		loginSlot:           make(chan struct{}, 1),
		failedLoginCooldown: time.Minute,
	}
	am.login = func() error {
		atomic.AddInt32(&logins, 1)
		<-release
		return errors.New("exit status 254")
	}

	errs := make([]error, callers)
	var wg sync.WaitGroup
	for i := range callers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = am.renew(context.Background(), 0)
		}(i)
	}
	// Let every caller reach the login slot before the first login fails.
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := atomic.LoadInt32(&logins); got != 1 {
		t.Errorf("want 1 login for %d queued callers after a failure, got %d", callers, got)
	}
	for i, err := range errs {
		if err == nil {
			t.Errorf("caller %d: want an error, got nil", i)
		}
	}
	if err := errs[callers-1]; err != nil && !strings.Contains(err.Error(), "exit status 254") {
		t.Errorf("a blocked caller should see why the last login failed, got: %v", err)
	}

	// Once the cooldown has passed, a later request may try again.
	am.lastLoginDone = time.Now().Add(-2 * time.Minute)
	am.login = func() error { atomic.AddInt32(&logins, 1); return nil }
	testAWSEnv(t)
	if err := am.renew(context.Background(), 0); err != nil {
		t.Fatalf("login after cooldown: %v", err)
	}
	if got := atomic.LoadInt32(&logins); got != 2 {
		t.Errorf("want a second login after the cooldown, got %d total", got)
	}
}

// The failure seen in the wild: login succeeds, then the very next call is
// rejected again. A second login cannot help, so requests must fail with an
// explanation instead of opening another browser tab each time.
func TestSuccessfulLoginThatDoesNotFixCredentialsIsNotRepeated(t *testing.T) {
	testAWSEnv(t)

	var logins, calls int32
	am := &AuthManager{
		profile:             "test",
		region:              "us-east-1",
		loginSlot:           make(chan struct{}, 1),
		futileLoginCooldown: 30 * time.Minute,
	}
	am.login = func() error { atomic.AddInt32(&logins, 1); return nil }
	rejected := func() error { atomic.AddInt32(&calls, 1); return roleAccessDenied() }

	// First request: one login, retried once, still rejected — and the error says
	// what to check rather than just "403".
	err := am.withRetry(context.Background(), rejected)
	if err == nil || !strings.Contains(err.Error(), "no assignment") {
		t.Fatalf("want an explained role-access error, got: %v", err)
	}
	if atomic.LoadInt32(&logins) != 1 || atomic.LoadInt32(&calls) != 2 {
		t.Fatalf("want 1 login and 2 attempts, got %d and %d", logins, calls)
	}

	// Every later request, however many, must not log in again.
	for range 5 {
		err = am.withRetry(context.Background(), rejected)
		if err == nil || !strings.Contains(err.Error(), "not starting another SSO login") {
			t.Fatalf("want the repeat login refused, got: %v", err)
		}
	}
	if got := atomic.LoadInt32(&logins); got != 1 {
		t.Errorf("want still 1 login after repeated failures, got %d", got)
	}
	// ...and the refusal does not consume an attempt on a second retry.
	if got := atomic.LoadInt32(&calls); got != 7 {
		t.Errorf("want 7 attempts in total (2 + 5), got %d", got)
	}

	// Credentials that start working again need no login at all.
	if err := am.withRetry(context.Background(), func() error { return nil }); err != nil {
		t.Errorf("working credentials must pass straight through: %v", err)
	}

	// A genuinely new expiry long after a good login is still renewed.
	am.lastLoginDone = time.Now().Add(-time.Hour)
	_ = am.withRetry(context.Background(), rejected)
	if got := atomic.LoadInt32(&logins); got != 2 {
		t.Errorf("want a login once the cooldown has passed, got %d total", got)
	}
}

// startupManager builds a manager whose startup credential check and login are
// scripted, pointing at a throwaway profile so config loading succeeds.
func startupManager(t *testing.T, retrieve func(call int32) error, login func(call int32) error) (*AuthManager, *int32, *int32) {
	t.Helper()
	testAWSEnv(t)
	var retrieves, logins int32
	am := &AuthManager{
		profile:             "test",
		region:              "us-east-1",
		loginSlot:           make(chan struct{}, 1),
		failedLoginCooldown: 2 * time.Minute,
		futileLoginCooldown: 30 * time.Minute,
	}
	am.retrieve = func(context.Context) error { return retrieve(atomic.AddInt32(&retrieves, 1)) }
	am.login = func() error { return login(atomic.AddInt32(&logins, 1)) }
	return am, &retrieves, &logins
}

// A startup login nobody approves must not take the proxy down: under launchd
// KeepAlive an exit meant a restart, which meant another 10-minute prompt, forever.
func TestStartupLoginFailureKeepsProxyUpAndFirstRequestRetries(t *testing.T) {
	am, _, logins := startupManager(t,
		func(int32) error { return &stubAPIError{code: "ExpiredTokenException"} },
		func(n int32) error {
			if n == 1 {
				return errors.New("exit status 254") // startup: nobody approved
			}
			return nil // the first request: approved this time
		})

	if err := am.start(context.Background()); err != nil {
		t.Fatalf("a failed startup login must not be fatal, got: %v", err)
	}
	if am.client == nil {
		t.Fatal("want a client so requests can be served")
	}
	if am.CredentialsReady() {
		t.Error("want CredentialsReady false after a failed startup login")
	}
	if !am.lastLoginDone.IsZero() {
		t.Error("a failed startup login must not start a cooldown: the first request should retry at once")
	}

	var attempts int32
	err := am.withRetry(context.Background(), func() error {
		if atomic.AddInt32(&attempts, 1) == 1 {
			return &stubAPIError{code: "ExpiredTokenException"}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("first request should recover via a fresh login, got: %v", err)
	}
	if got := atomic.LoadInt32(logins); got != 2 {
		t.Errorf("want 2 logins (startup + first request), got %d", got)
	}
}

func TestStartupLoginSuccessIsRecordedAndVerified(t *testing.T) {
	am, retrieves, logins := startupManager(t,
		func(n int32) error {
			if n == 1 {
				return &stubAPIError{code: "ExpiredTokenException"}
			}
			return nil // the post-login check passes
		},
		func(int32) error { return nil })

	if err := am.start(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !am.CredentialsReady() {
		t.Error("want CredentialsReady true after a login that produced working credentials")
	}
	if atomic.LoadInt32(logins) != 1 || atomic.LoadInt32(retrieves) != 2 {
		t.Errorf("want 1 login and 2 credential checks, got %d and %d", *logins, *retrieves)
	}
	if am.lastLoginDone.IsZero() || am.lastLoginErr != nil {
		t.Error("a successful startup login should be recorded so a failing first request is not met with another one")
	}
}

// Login succeeded but AWS still refuses: start anyway, report not ready, and
// leave the 30-minute window in force so requests do not prompt again.
func TestStartupLoginThatDoesNotFixCredentialsStillStarts(t *testing.T) {
	am, _, logins := startupManager(t,
		func(int32) error { return roleAccessDenied() },
		func(int32) error { return nil })

	if err := am.start(context.Background()); err != nil {
		t.Fatalf("want startup to continue, got: %v", err)
	}
	if am.CredentialsReady() {
		t.Error("want CredentialsReady false")
	}
	err := am.withRetry(context.Background(), func() error { return roleAccessDenied() })
	if err == nil || !strings.Contains(err.Error(), "not starting another SSO login") {
		t.Errorf("want the repeat login refused, got: %v", err)
	}
	if got := atomic.LoadInt32(logins); got != 1 {
		t.Errorf("want only the startup login, got %d", got)
	}
}

func TestStartupStillFailsOnNonSSOCredentialError(t *testing.T) {
	am, _, logins := startupManager(t,
		func(int32) error { return errors.New("no such profile") },
		func(int32) error { return nil })

	if err := am.start(context.Background()); err == nil {
		t.Fatal("a credential error that is not about SSO expiry should still be fatal")
	}
	if atomic.LoadInt32(logins) != 0 {
		t.Error("no login should be attempted for a non-SSO error")
	}
}
