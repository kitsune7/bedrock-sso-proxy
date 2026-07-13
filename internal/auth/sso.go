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

// Client returns the current Bedrock runtime client.
func (am *AuthManager) Client() *bedrockruntime.Client {
	am.mu.Lock()
	defer am.mu.Unlock()
	return am.client
}

// WithRetry executes fn with the current client. If fn returns an SSO credential
// error, it triggers re-authentication and retries fn exactly once.
func (am *AuthManager) WithRetry(ctx context.Context, fn func(*bedrockruntime.Client) error) error {
	client := am.Client()
	err := fn(client)
	if err == nil {
		return nil
	}

	if !isSSOError(err) {
		return err
	}

	// SSO expired — acquire lock and renew
	am.mu.Lock()
	defer am.mu.Unlock()

	log.Println("SSO session expired during request. Launching browser for re-authentication...")
	if loginErr := am.runSSOLogin(); loginErr != nil {
		return fmt.Errorf("SSO re-authentication failed: %w (original error: %v)", loginErr, err)
	}

	if loadErr := am.loadConfig(ctx); loadErr != nil {
		return fmt.Errorf("failed to reload AWS config after SSO login: %w", loadErr)
	}

	am.client = bedrockruntime.NewFromConfig(am.cfg)

	// Retry once with fresh credentials
	return fn(am.client)
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
