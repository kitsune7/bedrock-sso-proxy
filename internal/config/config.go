package config

import (
	"flag"
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	Port         int
	AWSProfile   string
	AWSRegion    string
	DefaultModel string
	CrossRegion  bool
	Verbose      bool
}

func Parse() *Config {
	_ = godotenv.Load()

	cfg := &Config{}

	flag.IntVar(&cfg.Port, "port", 8000, "listen port")
	flag.StringVar(&cfg.AWSProfile, "profile", "", "AWS SSO profile name")
	flag.StringVar(&cfg.AWSRegion, "region", "us-east-1", "AWS region")
	flag.StringVar(&cfg.DefaultModel, "default-model", "gpt-5.5", "default model when request omits it")
	flag.BoolVar(&cfg.CrossRegion, "cross-region", true, "prepend region prefix to model IDs for cross-region inference")
	flag.BoolVar(&cfg.Verbose, "verbose", false, "enable verbose logging")
	flag.Parse()

	// Environment variable overrides for flags not explicitly set
	explicitly := make(map[string]bool)
	flag.Visit(func(f *flag.Flag) { explicitly[f.Name] = true })

	if !explicitly["port"] {
		if v := os.Getenv("BEDROCK_PROXY_PORT"); v != "" {
			if p, err := strconv.Atoi(v); err == nil {
				cfg.Port = p
			}
		}
	}
	if !explicitly["profile"] {
		if v := os.Getenv("AWS_PROFILE"); v != "" {
			cfg.AWSProfile = v
		}
	}
	if !explicitly["region"] {
		if v := os.Getenv("AWS_REGION"); v != "" {
			cfg.AWSRegion = v
		}
	}
	if !explicitly["default-model"] {
		if v := os.Getenv("DEFAULT_MODEL"); v != "" {
			cfg.DefaultModel = v
		}
	}
	if !explicitly["cross-region"] {
		if v := os.Getenv("CROSS_REGION"); v != "" {
			cfg.CrossRegion = v == "true" || v == "1"
		}
	}
	if !explicitly["verbose"] {
		if v := os.Getenv("VERBOSE"); v != "" {
			cfg.Verbose = v == "true" || v == "1"
		}
	}

	return cfg
}
