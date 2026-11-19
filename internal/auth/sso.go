package auth

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go"
)

// Cooldowns on starting another `aws sso login` after one has finished. Without
// them every queued or retried request launched its own browser flow in turn,
// which a user experiences as an endless stream of prompts.
const (
	// A login that failed — typically nobody approved the device code, so the
	// CLI gave up after ~10 minutes — will fail again for the next request, and
	// for every request that queued behind it.
	defaultFailedLoginCooldown = 2 * time.Minute
	// A login that succeeded but left the credentials still rejected (for
	// example AWS answering GetRoleCredentials with 403 "No access" because the
	// signed-in identity has no assignment for the profile) cannot be repaired by
	// logging in again. Role credentials are refreshed from the SSO token without
	// a login, so a genuinely new expiry this soon after a good login is not a
	// real case. This only suppresses automatic logins: once access is fixed, the
	// next request picks up working credentials with no login at all.
	defaultFutileLoginCooldown = 30 * time.Minute
)

// AuthManager handles AWS credential loading and automatic SSO session renewal.
type AuthManager struct {
	// mu guards cfg/client/gen. Held only for field access — never across an
	// interactive `aws sso login`, or every in-flight request would block on the
	// browser round-trip.
	mu     sync.Mutex
	cfg    aws.Config
	client *bedrockruntime.Client
	// gen increments on every successful renewal, so a caller can tell whether
	// the credentials it failed with have since been replaced.
	gen uint64
	// loginSlot is a context-aware mutex (capacity 1) serializing SSO logins.
	loginSlot chan struct{}
	// login shells out to `aws sso login`; overridden in tests.
	login   func() error
	profile string
	region  string

	// lastLoginDone/lastLoginErr record how the most recent login ended
	// (lastLoginErr is nil when it succeeded). They are only touched while
	// holding loginSlot — or during NewManager, before the manager is shared.
	// The two cooldowns bound how soon another login may start after a failed or
	// a successful one respectively; zero disables that check.
	lastLoginDone       time.Time
	lastLoginErr        error
	failedLoginCooldown time.Duration
	futileLoginCooldown time.Duration

	// retrieve checks that the current config yields credentials; replaced in
	// tests. credsReady records the outcome at startup (see CredentialsReady).
	retrieve   func(context.Context) error
	credsReady bool
}

// NewManager creates an AuthManager, loading AWS config from the given SSO profile.
// If SSO credentials are expired, it will attempt to authenticate immediately.
//
// A startup login that fails (nobody approved it, say) is not fatal: the manager
// is returned anyway and the first request that needs credentials tries again.
// Exiting instead would hand the problem to a supervisor — launchd KeepAlive
// restarted the proxy after every failed 10-minute login, which opened another
// browser prompt each time, indefinitely.
func NewManager(profile, region string) (*AuthManager, error) {
	am := &AuthManager{
		profile:             profile,
		region:              region,
		loginSlot:           make(chan struct{}, 1),
		failedLoginCooldown: defaultFailedLoginCooldown,
		futileLoginCooldown: defaultFutileLoginCooldown,
	}
	am.login = am.runSSOLogin
	am.retrieve = func(ctx context.Context) error {
		_, err := am.cfg.Credentials.Retrieve(ctx)
		return err
	}
	if err := am.start(context.Background()); err != nil {
		return nil, err
	}
	return am, nil
}

// start loads the profile, validates its credentials, and builds the client.
func (am *AuthManager) start(ctx context.Context) error {
	if err := am.loadConfig(ctx); err != nil {
		// Try SSO login if initial load suggests expired credentials. Unlike the
		// credential check below, there is no usable config to fall back on if
		// this login fails, so that stays fatal.
		if !isSSOError(err) {
			return fmt.Errorf("failed to load AWS config: %w", err)
		}
		log.Println("SSO session not found or expired. Launching browser for authentication...")
		if loginErr := am.login(); loginErr != nil {
			return fmt.Errorf("SSO login failed: %w (original error: %v)", loginErr, err)
		}
		am.finishLogin(nil)
		if err := am.loadConfig(ctx); err != nil {
			return fmt.Errorf("failed to load AWS config after SSO login: %w", err)
		}
	}

	// Validate credentials by attempting to retrieve them
	am.credsReady = true
	if err := am.retrieve(ctx); err != nil {
		if !isSSOError(err) {
			return fmt.Errorf("failed to retrieve AWS credentials: %w", err)
		}
		am.credsReady = false
		log.Println("SSO session expired. Launching browser for authentication...")
		if loginErr := am.login(); loginErr != nil {
			// Deliberately not recorded as a finished login: the first request
			// should try again straight away rather than wait out a cooldown for
			// a login nobody was around to approve.
			log.Printf("SSO login failed: %v", loginErr)
			log.Println("Continuing without credentials; the first request that needs them will retry the login.")
		} else {
			am.finishLogin(nil)
			if err := am.loadConfig(ctx); err != nil {
				return fmt.Errorf("failed to reload AWS config after SSO login: %w", err)
			}
			// The login succeeded; check it actually produced working credentials.
			if err := am.retrieve(ctx); err != nil {
				log.Printf("Warning: credentials still rejected after SSO login: %v", am.annotate(err))
			} else {
				am.credsReady = true
			}
		}
	}

	am.client = bedrockruntime.NewFromConfig(am.cfg)
	return nil
}

// CredentialsReady reports whether credentials were confirmed working at
// startup. When false the manager still serves requests, renewing the SSO login
// on the first one that needs it.
func (am *AuthManager) CredentialsReady() bool { return am.credsReady }

// NewManagerFromConfig builds an AuthManager around an existing AWS config,
// skipping profile loading and SSO. Renewal is unavailable — there is no
// profile to log in to — so this is for tests and for callers that already hold
// credentials.
func NewManagerFromConfig(cfg aws.Config) *AuthManager {
	am := &AuthManager{
		cfg:       cfg,
		client:    bedrockruntime.NewFromConfig(cfg),
		loginSlot: make(chan struct{}, 1),
	}
	am.login = am.runSSOLogin
	return am
}

// Client returns the current Bedrock runtime client.
func (am *AuthManager) Client() *bedrockruntime.Client {
	am.mu.Lock()
	defer am.mu.Unlock()
	return am.client
}

// Config returns the current AWS config. Callers that talk to an endpoint the
// pinned SDK has no client for (bedrock-mantle) sign requests from it directly.
func (am *AuthManager) Config() aws.Config {
	am.mu.Lock()
	defer am.mu.Unlock()
	return am.cfg
}

// WithRetry executes fn with the current client. If fn returns an SSO credential
// error, it triggers re-authentication and retries fn exactly once.
func (am *AuthManager) WithRetry(ctx context.Context, fn func(*bedrockruntime.Client) error) error {
	return am.withRetry(ctx, func() error { return fn(am.Client()) })
}

// WithRetryConfig is WithRetry for backends reached without an SDK client — it
// hands fn the AWS config so it can sign its own requests.
func (am *AuthManager) WithRetryConfig(ctx context.Context, fn func(aws.Config) error) error {
	return am.withRetry(ctx, func() error { return fn(am.Config()) })
}

func (am *AuthManager) withRetry(ctx context.Context, fn func() error) error {
	// Snapshot the generation *before* the call so renew can tell whether the
	// credentials this attempt failed with are the current ones.
	gen := am.generation()
	err := fn()
	if err == nil || !isSSOError(err) {
		return am.annotate(err)
	}
	if am.profile == "" {
		// Nothing to log in to — see NewManagerFromConfig.
		return err
	}
	if renewErr := am.renew(ctx, gen); renewErr != nil {
		return fmt.Errorf("%w (original error: %v)", renewErr, err)
	}
	// Retry once with fresh credentials.
	return am.annotate(fn())
}

// annotate explains a role-access denial that a login did not fix, since the
// raw "403 No access" says nothing about what to check. It returns err
// unchanged for every other case, including nil.
func (am *AuthManager) annotate(err error) error {
	if !isRoleAccessDenied(err) {
		return err
	}
	return fmt.Errorf("%w (AWS denied role credentials for profile %q even though the SSO login succeeded; "+
		"the signed-in identity probably has no assignment for the profile's account and permission set)",
		err, am.profile)
}

func (am *AuthManager) generation() uint64 {
	am.mu.Lock()
	defer am.mu.Unlock()
	return am.gen
}

// renew re-authenticates and rebuilds the config and client.
//
// It is single-flight on gen: a caller whose generation is already stale skips
// the login outright, because a concurrent renew has just completed one. Without
// that check, N requests failing on the same expired token each spawn their own
// `aws sso login`, and every device code but the one the user actually approved
// dies with InvalidGrantException — which fails those requests, which makes the
// client retry, which spawns more logins. That loop is unbreakable by approving.
func (am *AuthManager) renew(ctx context.Context, gen uint64) error {
	// Context-aware lock: a caller that gives up while another login is in
	// flight leaves rather than piling onto the browser queue.
	select {
	case am.loginSlot <- struct{}{}:
		defer func() { <-am.loginSlot }()
	case <-ctx.Done():
		return ctx.Err()
	}

	if am.generation() != gen {
		// Another request already logged in. Our caller retries against it.
		return nil
	}

	// gen is still current, so no login has repaired the credentials this caller
	// failed with. If one just ended anyway, a second will not do better: either
	// it failed (and every request queued behind it would fail the same way, one
	// 10-minute browser wait at a time), or it succeeded and the credentials were
	// still rejected. Fail fast and let a later request, after the cooldown, try.
	if !am.lastLoginDone.IsZero() {
		since := time.Since(am.lastLoginDone)
		ago := since.Round(time.Second)
		if am.lastLoginErr != nil && since < am.failedLoginCooldown {
			return fmt.Errorf("not starting another SSO login: the last attempt %s ago failed: %w", ago, am.lastLoginErr)
		}
		if am.lastLoginErr == nil && since < am.futileLoginCooldown {
			return fmt.Errorf("not starting another SSO login: credentials were still rejected %s after a successful login; "+
				"profile %q may lack access to its account or permission set (run `aws sso login --profile %s` yourself once that is fixed)",
				ago, am.profile, am.profile)
		}
	}

	log.Println("SSO session expired during request. Launching browser for re-authentication...")
	if err := am.login(); err != nil {
		err = fmt.Errorf("SSO re-authentication failed: %w", err)
		am.finishLogin(err)
		return err
	}
	// Deliberately not ctx: the login and reload outlive the request that
	// triggered them. Tying them to a request context meant a client that hung
	// up mid-browser-flow killed the login for everyone waiting on it.
	cfg, err := am.newConfig(context.Background())
	if err != nil {
		err = fmt.Errorf("failed to reload AWS config after SSO login: %w", err)
		am.finishLogin(err)
		return err
	}
	am.finishLogin(nil)

	am.mu.Lock()
	defer am.mu.Unlock()
	am.cfg = cfg
	am.client = bedrockruntime.NewFromConfig(cfg)
	am.gen++
	return nil
}

// finishLogin records how a login ended. The caller must hold loginSlot.
func (am *AuthManager) finishLogin(err error) {
	am.lastLoginDone = time.Now()
	am.lastLoginErr = err
}

func (am *AuthManager) newConfig(ctx context.Context) (aws.Config, error) {
	return awsconfig.LoadDefaultConfig(ctx,
		awsconfig.WithSharedConfigProfile(am.profile),
		awsconfig.WithRegion(am.region),
	)
}

func (am *AuthManager) loadConfig(ctx context.Context) error {
	cfg, err := am.newConfig(ctx)
	if err != nil {
		return err
	}
	am.cfg = cfg
	return nil
}

func (am *AuthManager) runSSOLogin() error {
	// ponytail: no timeout of our own — the AWS CLI already bounds its device-code
	// poll (~10 min). Add one here only if that window proves too long to wait.
	cmd := exec.Command("aws", "sso", "login", "--profile", am.profile)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// isRoleAccessDenied reports whether err is the SSO portal refusing to issue
// role credentials (GetRoleCredentials → 403 ForbiddenException "No access").
// That usually means the signed-in identity has no assignment for the
// profile's account and permission set, which a fresh login cannot repair.
func isRoleAccessDenied(err error) bool {
	if err == nil {
		return false
	}
	// Matched on the message: the SDK's generic API error type for this response
	// is not one callers can rely on reaching through errors.As.
	msg := err.Error()
	return strings.Contains(msg, "GetRoleCredentials") && strings.Contains(msg, "ForbiddenException")
}

// isSSOError checks if an error is related to expired or missing SSO credentials.
func isSSOError(err error) bool {
	if err == nil {
		return false
	}

	// Check for smithy API errors
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		code := apiErr.ErrorCode()
		switch code {
		// "UnauthorizedException" is what the SSO portal returns for a missing
		// or expired access token.
		case "ExpiredTokenException",
			"UnrecognizedClientException",
			"InvalidIdentityToken",
			"ExpiredToken",
			"UnauthorizedException":
			return true
		// Authorization or throttling, not authentication. Without this case the
		// message check below would match the SDK's "failed to refresh cached
		// credentials" wrapper and open a browser for a throttled call.
		// AccessDeniedException used to do exactly that whenever a model was
		// ungranted. (The SSO portal's ForbiddenException is deliberately not
		// here: it can follow a stale token, so one login is worth trying — the
		// cooldowns in renew stop it looping when the login does not help.)
		case "AccessDeniedException",
			"TooManyRequestsException",
			"ThrottlingException":
			return false
		}
	}

	// Check error message strings as fallback. Keep these specific to token
	// expiry — "The SSO" and "sso login" also matched our own renewal-failure
	// text, so a failed login classified as "needs a login".
	msg := err.Error()
	ssoIndicators := []string{
		"SSO session",
		"token has expired",
		"InvalidIdentityToken",
		"expired SSO",
		"refresh_token",
		"failed to refresh cached credentials",
		"no cached credentials",
	}
	for _, indicator := range ssoIndicators {
		if strings.Contains(strings.ToLower(msg), strings.ToLower(indicator)) {
			return true
		}
	}

	return false
}
