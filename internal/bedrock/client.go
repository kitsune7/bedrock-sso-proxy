package bedrock

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// Converse calls the Bedrock Converse API.
func Converse(ctx context.Context, client *bedrockruntime.Client, input *bedrockruntime.ConverseInput) (*bedrockruntime.ConverseOutput, error) {
	return client.Converse(ctx, input)
}

// ConverseStream calls the Bedrock ConverseStream API.
func ConverseStream(ctx context.Context, client *bedrockruntime.Client, input *bedrockruntime.ConverseStreamInput) (*bedrockruntime.ConverseStreamOutput, error) {
	return client.ConverseStream(ctx, input)
}
