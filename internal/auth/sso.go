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
	mu      sync.Mutex
	cfg     aws.Config
	client  *bedrockruntime.Client
	profile string
	region  string
}

// NewManager creates an AuthManager, loading AWS config from the given SSO profile.
// If SSO credentials are expired, it will attempt to authenticate immediately.
func NewManager(profile, region string) (*AuthManager, error) {
	am := &AuthManager{
		profile: profile,
		region:  region,
	}

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
	return &AuthManager{
		cfg:    cfg,
		client: bedrockruntime.NewFromConfig(cfg),
	}
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
	err := fn()
	if err == nil || !isSSOError(err) {
		return err
	}
	if am.profile == "" {
		// Nothing to log in to — see NewManagerFromConfig.
		return err
	}
	if renewErr := am.renew(ctx); renewErr != nil {
		return fmt.Errorf("%w (original error: %v)", renewErr, err)
	}
	// Retry once with fresh credentials.
	return fn()
}

// renew re-authenticates and rebuilds the config and client. fn must be called
// outside this lock — the accessors take it too, and it is not reentrant.
func (am *AuthManager) renew(ctx context.Context) error {
	am.mu.Lock()
	defer am.mu.Unlock()

	log.Println("SSO session expired during request. Launching browser for re-authentication...")
	if err := am.runSSOLogin(); err != nil {
		return fmt.Errorf("SSO re-authentication failed: %w", err)
	}
	if err := am.loadConfig(ctx); err != nil {
		return fmt.Errorf("failed to reload AWS config after SSO login: %w", err)
	}
	am.client = bedrockruntime.NewFromConfig(am.cfg)
	return nil
}

func (am *AuthManager) loadConfig(ctx context.Context) error {
	cfg, err := awsconfig.LoadDefaultConfig(ctx,
		awsconfig.WithSharedConfigProfile(am.profile),
		awsconfig.WithRegion(am.region),
	)
	if err != nil {
		return err
	}
	am.cfg = cfg
	return nil
}

func (am *AuthManager) runSSOLogin() error {
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
		case "ExpiredTokenException",
			"UnrecognizedClientException",
			"InvalidIdentityToken",
			"ExpiredToken",
			"AccessDeniedException":
			return true
		}
	}

	// Check error message strings as fallback
	msg := err.Error()
	ssoIndicators := []string{
		"SSO session",
		"token has expired",
		"InvalidIdentityToken",
		"expired SSO",
		"refresh_token",
		"sso login",
		"The SSO",
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
