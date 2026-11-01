package telemetry;

import java.util.ArrayList;
import java.util.List;

/**
 * Timing and token data for ONE document, collected by LlamaVascularExtractor on
 * the thread that processes the document, then handed to the caller with
 * extractor.takeTelemetry(path). Covers every model call made for the document
 * (primary, larger-budget retry, JSON repair), summed.
 *
 *   wallMs                  whole document, including reading the file and parsing
 *   requestMs               time inside HTTP calls to llama-server (queueing for a
 *                           slot, prompt processing and generation)
 *   promptTokens/completionTokens   server-reported usage, summed over calls
 *   promptN/promptMs        llama-server's own timing of prompt processing (tokens
 *                           actually evaluated, i.e. not served from the prompt cache)
 *   predictedN/predictedMs  llama-server's own timing of generation
 *   cachedTokens            prompt tokens reused from the cache (system prompt prefix)
 *
 * The promptN/promptMs and predictedN/predictedMs pairs give tokens per second
 * for one request, independent of how many documents run in parallel, which is
 * what to compare between computers. startEpochMs/endEpochMs let concurrency be
 * reconstructed afterwards.
 *
 * Also holds the raw model output of every call (rawResponses) so a parse or
 * repair failure can be told apart from an extraction error.
 */
public final class DocumentTelemetry {

    /** One model call's raw text and headline numbers. */
    public static final class RawResponse {
        public final String stage;          // primary, retry or repair
        public final String finishReason;
        public final int promptTokens;
        public final int completionTokens;
        public final long requestMs;
        public final String content;
        public final String reasoning;

        public RawResponse(String stage, String finishReason, int promptTokens,
                int completionTokens, long requestMs, String content, String reasoning) {
            this.stage = stage;
            this.finishReason = finishReason;
            this.promptTokens = promptTokens;
            this.completionTokens = completionTokens;
            this.requestMs = requestMs;
            this.content = content;
            this.reasoning = reasoning;
        }
    }

    public long startEpochMs;
    public long endEpochMs;
    public long wallMs;
    public long requestMs;
    public int modelCalls;
    public long promptTokens;
    public long completionTokens;
    public long promptN;
    public double promptMs;
    public long predictedN;
    public double predictedMs;
    public long cachedTokens;
    public final List<RawResponse> rawResponses = new ArrayList<>();

    /** Prompt-processing speed (tokens/second) for this document, or -1 if the server gave no timings. */
    public double promptTokensPerSecond() {
        return promptMs > 0 ? promptN * 1000.0 / promptMs : -1;
    }

    /** Generation speed (tokens/second) for this document, or -1 if the server gave no timings. */
    public double generationTokensPerSecond() {
        return predictedMs > 0 ? predictedN * 1000.0 / predictedMs : -1;
    }
}
