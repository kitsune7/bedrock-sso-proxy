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

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/smithy-go"
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
}

// NewManager creates an AuthManager, loading AWS config from the given SSO profile.
// If SSO credentials are expired, it will attempt to authenticate immediately.
func NewManager(profile, region string) (*AuthManager, error) {
	am := &AuthManager{
		profile:   profile,
		region:    region,
		loginSlot: make(chan struct{}, 1),
	}
	am.login = am.runSSOLogin

	if err := am.loadConfig(context.Background()); err != nil {
		// Try SSO login if initial load suggests expired credentials
		if isSSOError(err) {
			log.Println("SSO session not found or expired. Launching browser for authentication...")
			if loginErr := am.runSSOLogin(); loginErr != nil {
				return nil, fmt.Errorf("SSO login failed: %w (original error: %v)", loginErr, err)
			}
			if err := am.loadConfig(context.Background()); err != nil {
				return nil, fmt.Errorf("failed to load AWS config after SSO login: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to load AWS config: %w", err)
		}
	}

	// Validate credentials by attempting to retrieve them
	if _, err := am.cfg.Credentials.Retrieve(context.Background()); err != nil {
		if isSSOError(err) {
			log.Println("SSO session expired. Launching browser for authentication...")
			if loginErr := am.runSSOLogin(); loginErr != nil {
				return nil, fmt.Errorf("SSO login failed: %w", loginErr)
			}
			if err := am.loadConfig(context.Background()); err != nil {
				return nil, fmt.Errorf("failed to reload AWS config after SSO login: %w", err)
			}
		} else {
			return nil, fmt.Errorf("failed to retrieve AWS credentials: %w", err)
		}
	}

	am.client = bedrockruntime.NewFromConfig(am.cfg)
	return am, nil
}

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
		return err
	}
	if am.profile == "" {
		// Nothing to log in to — see NewManagerFromConfig.
		return err
	}
	if renewErr := am.renew(ctx, gen); renewErr != nil {
		return fmt.Errorf("%w (original error: %v)", renewErr, err)
	}
	// Retry once with fresh credentials.
	return fn()
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

	log.Println("SSO session expired during request. Launching browser for re-authentication...")
	if err := am.login(); err != nil {
		return fmt.Errorf("SSO re-authentication failed: %w", err)
	}
	// Deliberately not ctx: the login and reload outlive the request that
	// triggered them. Tying them to a request context meant a client that hung
	// up mid-browser-flow killed the login for everyone waiting on it.
	cfg, err := am.newConfig(context.Background())
	if err != nil {
		return fmt.Errorf("failed to reload AWS config after SSO login: %w", err)
	}

	am.mu.Lock()
	defer am.mu.Unlock()
	am.cfg = cfg
	am.client = bedrockruntime.NewFromConfig(cfg)
	am.gen++
	return nil
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
		// AccessDeniedException is deliberately absent: it means authorization,
		// not authentication. Treating it as an expired session opened a browser
		// every time a model was ungranted or a call was throttled.
		case "ExpiredTokenException",
			"UnrecognizedClientException",
			"InvalidIdentityToken",
			"ExpiredToken":
			return true
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
